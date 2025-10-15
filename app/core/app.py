import json

import chainlit as cl
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.core.prompts.query_normalization_prompt import \
    QUERY_NORMALIZATION_PROMPT
from app.core.prompts.rag_prompt import RAG_PROMPT
from app.core.services.similarity import SimilarityConverter
from app.data.embedding import EmbeddingFactory
from app.data.faiss_index import FAISSIndexManager

main_llm = ChatOpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL,
    model=settings.QUERY_MODEL,
    temperature=settings.QUERY_MODEL_TEMPERATURE,
    streaming=True,
)

normalization_llm = ChatOpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL,
    model=settings.QUERY_MODEL,
    temperature=settings.QUERY_MODEL_TEMPERATURE,
    streaming=True,
)


async def init_conversation():
    """Initialize the chat session."""
    # initialize memory
    memory = ConversationBufferWindowMemory(
        k=settings.MEMORY_WINDOW, return_messages=True, memory_key="history"
    )
    cl.user_session.set("memory", memory)

    await cl.Message(
        content="嗨您好，我是您的 AI 助手！ \n我今天可以幫助您處理哪些相關的事情呢？"
    ).send()


async def check_and_rewrite_query(query: str, history: list) -> dict:
    """rewrite check"""
    history_text = "\n".join(
        [
            f"{'用戶' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
            # last 5 interactions (1 interaction = HumanMessage + AIMessage)
            for msg in history[-(settings.MEMORY_WINDOW * 2) :]
        ]
    )

    prompt = QUERY_NORMALIZATION_PROMPT.format(history=history_text, query=query)
    response = await normalization_llm.ainvoke([HumanMessage(content=prompt)])

    try:
        print("Trying to parse JSON...")
        result = json.loads(response.content)
        return result
    except json.JSONDecodeError:
        print("JSON parsing failed.")
        return {
            "needs_rewrite": False,
            "reason": "JSON parsing failed",
            "rewritten_query": query,
        }


@cl.on_message
async def handle_user_message(message: cl.Message):
    """
    Process a user message from the Chainlit UI and generate a response.
    """

    memory = cl.user_session.get("memory")

    # Step 1: Query Rewriting Check
    rewrite_msg = cl.Message(content="🔍 分析查詢中...")
    await rewrite_msg.send()

    history = memory.load_memory_variables({}).get("history", [])
    print("\n")
    print("msg：", message.content)
    print("history：", history)
    rewrite_result = await check_and_rewrite_query(message.content, history)

    if rewrite_result["needs_rewrite"]:
        rewrite_msg.content = f"✏️ **查詢已優化**\n原始: {message.content}\n優化: {rewrite_result['rewritten_query']}\n原因: {rewrite_result['reason']}"
        final_query = rewrite_result["rewritten_query"]
    else:
        rewrite_msg.content = "✓ 查詢清晰，無需重寫"
        final_query = message.content

    await rewrite_msg.update()

    # Step 2: Retrieve Relevant Documents
    retrieval_msg = cl.Message(content="📚 檢索相關文件...")
    await retrieval_msg.send()
    await cl.sleep(1)  # make sure the msg show up

    embedding_model = EmbeddingFactory.create_jina_embedding_model()
    vector_index = FAISSIndexManager.load(settings.FAISS_INDEX_PATH, embedding_model, settings.FAISS_INDEX_NAME)
    docs_with_scores = vector_index.similarity_search_with_score(
        final_query, k=settings.RETRIEVAL_TOP_K
    )

    if docs_with_scores:
        contexts = []
        for i, (doc, score) in enumerate(docs_with_scores):
            score = SimilarityConverter.score_to_similarity(score)
            if score >= settings.SIMILARITY_THRESHOLD:
                print("\n")
                print(f"找到的問題: {doc.page_content}")
                print(f"對應答案: {doc.metadata['answer']}")
                print(f"相似度分數: {score}")
                contexts.append(
                    f"[片段 {i+1}]\n 問題： {doc.page_content}\n 回答： {doc.metadata['answer']}"
                )
            else:
                break

        if contexts:
            context = "\n---\n".join(contexts)
        else:
            context = "無相關文件內容。"

        retrieval_msg.content = f"✓ 找到 {len(docs_with_scores)} 個相關片段"
    else:
        context = "無相關文件內容。"
        retrieval_msg.content = "⚠️ 未找到相關文件，僅依靠對話歷史回答"
    await retrieval_msg.update()

    # Step 3: Generate Answer
    answer_msg = cl.Message(content="")
    await answer_msg.send()

    if history:
        history_text = "\n".join(
            [
                f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: \n{msg.content}"
                for msg in history[-(settings.MEMORY_WINDOW * 2) :]
            ]
        )
    else:
        history_text = "暫時無對話歷史。"

    prompt = RAG_PROMPT.format(
        history=history_text, context=context, question=final_query
    )
    print("Final prompt to LLM:", prompt)

    # answer with streaming
    full_response = ""
    async for chunk in main_llm.astream([HumanMessage(content=prompt)]):
        if chunk.content:
            full_response += chunk.content
            await answer_msg.stream_token(chunk.content)

    await answer_msg.update()

    # Step 4: Display Sources
    if contexts:
        sources = []
        for i, (doc, score) in enumerate(docs_with_scores):
            score = SimilarityConverter.score_to_similarity(score)
            if score >= settings.SIMILARITY_THRESHOLD:
                # TODO: limit `doc.metadata['answer']` to prevent overly long source display
                sources.append(
                    f"**來源 {i+1}** (來自 {settings.QA_FILE_PATH}):\n> {doc.page_content}\n> {doc.metadata['answer']}"
                )
    else:
        sources = ["無相關來源文件。"]

    source_content = "**具體來源請參考以下：**\n" + "\n\n".join(sources)
    source_msg = cl.Message(content=source_content)
    await source_msg.send()

    # Step 5: Update Memory
    full_response = (
        full_response[-5:] if full_response.endswith("---\n\n") else full_response
    )
    memory.save_context(
        {"input": message.content},
        {"output": full_response + "\n\n" + source_content + "\n---\n"},
    )

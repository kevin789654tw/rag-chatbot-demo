import chainlit as cl
from langchain.schema import HumanMessage

from app.config import settings
from app.core.prompts.rag_prompt import RAG_PROMPT
from app.core.services.llm_provider import LLMFactory
from app.core.services.memory import SessionMemory
from app.core.services.query_normalizer import QueryNormalizer
from app.core.services.rag_service import RAGService
from app.data.embedding import EmbeddingFactory
from app.data.faiss_index import FAISSIndexManager


async def init_conversation():
    """Initialize the chat session."""
    # initialize memory
    memory = SessionMemory.create(settings.MEMORY_WINDOW)
    cl.user_session.set("memory", memory)

    await cl.Message(
        content=(
            "Hi! I'm your AI Assistant 😊\n" "What would you like help with today?"
        )
    ).send()


@cl.on_message
async def handle_user_message(message: cl.Message) -> None:
    """Process a user message from the Chainlit UI, normalize query, retrieve docs,
    generate response, and update memory.

    Returns:
        None
    """

    memory = cl.user_session.get("memory")

    llm_manager = LLMFactory()
    llm = llm_manager.get_llm(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        model=settings.QUERY_MODEL,
        temperature=settings.QUERY_MODEL_TEMPERATURE,
    )  # use cache to ensure the same config returns the same LLM instance

    # Step 1: Query Normalization
    rewrite_msg = cl.Message(content="🔍 Let me analyze your question...")
    await rewrite_msg.send()

    query_normalizer = QueryNormalizer(llm)
    history = memory.load_memory_variables({}).get("history", [])
    rewrite_result = await query_normalizer.normalize_and_rewrite(
        message.content, history
    )
    final_queries = rewrite_result["normalize_query"]

    if rewrite_result["needs_rewrite"]:
        rewrite_msg.content = (
            f"✏️ **Query Normalized** \n"
            f"• Your original query: {message.content} \n"
            f"• Normalized query: {rewrite_result['normalize_query']} \n"
        )
    else:
        rewrite_msg.content = (
            "✅ Got it! Your question is clear and doesn’t need rewriting."
        )

    await rewrite_msg.update()

    # Step 2: Retrieve Documents
    retrieval_msg = cl.Message(content="📚 Let me fetch the relevant documents...")
    await retrieval_msg.send()
    await cl.sleep(1)  # make sure the msg show up

    embedding_model = EmbeddingFactory.create_jina_embedding_model()
    vector_index = FAISSIndexManager.load(
        settings.FAISS_INDEX_PATH, embedding_model, settings.FAISS_INDEX_NAME
    )

    rag_service = RAGService()
    docs_with_scores = rag_service.retrieve_docs(
        final_queries, vector_index, settings.RETRIEVAL_TOP_K
    )

    if docs_with_scores:
        context, sources = rag_service.build_contexts_and_sources(
            docs_with_scores,
            settings.SIMILARITY_THRESHOLD,
            source_type=settings.SOURCE_TYPE,
            dataset_name=settings.DATASET_NAME,
        )
        retrieval_msg.content = f"✅ I found {len(docs_with_scores)} relevant snippets!"
    else:
        context, sources = "No relevant documents found.", ""
        retrieval_msg.content = (
            "⚠️ I couldn't find any relevant documents. "
            "I’ll answer based on our chat history."
        )

    await retrieval_msg.update()

    # Step 3: Generate Response
    res_msg = cl.Message(content="")
    await res_msg.send()

    if history:
        history_text = "\n".join(
            f"{'#### User' if isinstance(msg, HumanMessage) else '#### AI'}: \n{msg.content}"
            for msg in history
        )
    else:
        history_text = "No conversation history." + "\n\n---"

    prompt = RAG_PROMPT.format(
        history=history_text, context=context, question=final_queries
    )

    # response with streaming
    full_response = ""
    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        if chunk.content:
            full_response += chunk.content
            await res_msg.stream_token(chunk.content)

    await res_msg.update()

    # Step 4: Display Sources
    source_msg = cl.Message(content=sources)
    await source_msg.send()

    # Step 5: Update Memory
    full_response = (
        full_response[-5:] if full_response.endswith("---\n\n") else full_response
    )  # prevent unexpected '---' in the markdown format
    output_content = f"{full_response}\n{sources}\n\n---"
    memory.save_context(
        {"input": message.content},
        {"output": output_content},
    )

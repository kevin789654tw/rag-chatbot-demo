# import pandas as pd
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import settings


def score_to_similarity(distance: float):
    return 1 - (distance**2) / 2


def similarity_to_score(cosine_similarity: float):
    return (2 * (1 - cosine_similarity)) ** 0.5


qa_dataset = load_dataset(
    settings.DATASET, revision=settings.DATASET_HASH
)  # nosec: B615

questions = list(qa_dataset["train"][settings.QUES_COLUMN])
answers = list(qa_dataset["train"][settings.ANS_COLUMN])


# # --- Load local CSV file ---
# df = pd.read_csv(str(settings.QA_FILE_PATH)).dropna()  # remove empty rows

# if df.empty or len(df.columns) < 2:
#     raise ValueError("CSV data is empty or doesn't have at least 2 columns.")

# questions = df.iloc[:, 0].tolist()
# answers = df.iloc[:, 1].tolist()
# # ---------------------------

ques_pattern, ans_pattern = settings.QUES_PATTERN, settings.ANS_PATTERN

documents = []
for question, answer in zip(questions[0:500], answers[0:500]):

    if ques_pattern and ques_pattern in question:
        question = question.split(ques_pattern)[1].strip()

    if ans_pattern and ans_pattern in answer:
        answer = answer.split(ans_pattern)[1].strip()

    doc = Document(page_content=question, metadata={"answer": answer})
    documents.append(doc)

embeddings = JinaEmbeddings(
    jina_api_key=settings.JINA_API_KEY, model_name=settings.EMBEDDING_MODEL
)
vector_index = FAISS.from_documents(documents, embeddings)

# save
vector_index.save_local(settings.FAISS_INDEX_PATH)

query = "How many types of Eucalyptus are grown around the world?"
docs_with_scores = vector_index.similarity_search_with_score(query, k=3)

threshold = 0

for doc, score in docs_with_scores:
    score = score_to_similarity(score)
    if score >= threshold:
        print("\n")
        print("Found Question:", doc.page_content)
        print("Corresponding Answer:", doc.metadata["answer"])
        print("Similarity Score:", score)
    else:
        print("\n")
        print("No relevant question found.")

if not docs_with_scores:
    print("No relevant question found.")

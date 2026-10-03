"""显式建立固定语料索引；启动/提问绝不偷偷下载或重建向量。"""

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from power_forecast_service.assistant.retrieval import corpus, embed, EMBEDDING_REVISION
from power_forecast_service.settings import Settings
from power_forecast_service.assistant.storage import AssistantChunk


def main():
    pack = corpus()
    engine = create_engine(Settings.from_environment().database_url)
    vectors = embed([c["title"] + "\n" + c["text"] for c in pack["chunks"]])
    with Session(engine) as session, session.begin():
        for chunk, vector in zip(pack["chunks"], vectors, strict=True):
            old = session.get(AssistantChunk, chunk["id"])
            values = dict(corpus_sha256=pack["sha256"], text_sha256=chunk["text_sha256"],
                          embedding_revision=EMBEDDING_REVISION, content=chunk["text"], embedding=vector)
            if old:
                if old.text_sha256 != chunk["text_sha256"] or old.content != chunk["text"]:
                    raise ValueError("document identity collision")
                for key, value in values.items():
                    setattr(old, key, value)
            else:
                session.add(AssistantChunk(id=chunk["id"], **values))
    print(f"Indexed {len(vectors)} fixed-revision documents; exact cosine search, no ANN index.")
    engine.dispose()


if __name__ == "__main__":
    main()

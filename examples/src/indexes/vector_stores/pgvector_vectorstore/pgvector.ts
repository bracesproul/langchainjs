import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PGVectorStore } from "langchain/vectorstores/pgvector";
import { PoolConfig } from "pg";

// First, follow set-up instructions at
// https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/pgvector

export const run = async () => {
  const config = {
    postgresConnectionOptions: {
      type: "postgres",
      host: "127.0.0.1",
      port: 5433,
      user: "admin",
      password: "admin",
      database: "test",
    } as PoolConfig,
    tableName: "testlangchain",
    idColumnName: "id",
    vectorColumnName: "vector",
    contentColumnName: "content",
    metadataColumnName: "metadata",
  };

  const pgvectorStore = await PGVectorStore.initialize(
    new OpenAIEmbeddings(),
    config
  );

  await pgvectorStore.addDocuments([
    { pageContent: "what's this", metadata: { a: 2 } },
    { pageContent: "Cat drinks milk", metadata: { a: 1 } },
  ]);

  const results = await pgvectorStore.similaritySearch("hello", 2);

  console.log(results);
};

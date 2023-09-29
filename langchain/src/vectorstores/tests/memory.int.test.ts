import { test, expect } from "@jest/globals";

import { OpenAIEmbeddings } from "../../embeddings/openai.js";
import { Document } from "../../document.js";
import { MemorySemanticVectorStore, MemoryVectorStore } from "../memory.js";

test.skip("MemoryVectorStore with external ids", async () => {
  const embeddings = new OpenAIEmbeddings();

  const store = new MemoryVectorStore(embeddings);

  expect(store).toBeDefined();

  await store.addDocuments([
    { pageContent: "hello", metadata: { a: 1 } },
    { pageContent: "hi", metadata: { a: 1 } },
    { pageContent: "bye", metadata: { a: 1 } },
    { pageContent: "what's this", metadata: { a: 1 } },
  ]);

  const results = await store.similaritySearch("hello", 1);

  expect(results).toHaveLength(1);

  expect(results).toEqual([
    new Document({ metadata: { a: 1 }, pageContent: "hello" }),
  ]);
});

describe("MemoryVectorStore", () => {
  const embeddings = new OpenAIEmbeddings();
  const semanticCache = new MemorySemanticVectorStore(embeddings);
  const llmKey = "llm:key";

  afterEach(async () => {
    await semanticCache.clear(llmKey);
  });

  test("can perform an update with new generations", async () => {
    const prompt = "Who killed John F. Kennedy?";
    const generation = {
      text: "Lee Harvey Oswald",
    };

    await semanticCache.update(prompt, llmKey, [generation]);

    // Check with the exact same prompt. This test is not checking
    // similarity search, but rather that the cache is working.
    const results = await semanticCache.lookup(prompt, llmKey);
    expect(results).toHaveLength(1);
    expect(results).toEqual(
      expect.arrayContaining([expect.objectContaining(generation)])
    );
  });

  test("can perform a semantic search cache lookup", async () => {
    const initialPrompt = "Who killed John F. Kennedy?";
    const searchPrompt = "Who was John F. Kennedy's murderer?";
    const initialGeneration = {
      text: "Lee Harvey Oswald",
    };
    const codeSearchPrompt = "Is TypeScript coding better than Python coding?";
    const initialCodeGeneration = {
      text: "Yes, TypeScript is better than Python.",
    };

    // Add two to ensure it's not just returning the same generation.
    await semanticCache.update(initialPrompt, llmKey, [initialGeneration]);
    await semanticCache.update("Is TypeScript better than Python?", llmKey, [initialCodeGeneration]);

    const results = await semanticCache.lookup(searchPrompt, llmKey);
    const codeResults = await semanticCache.lookup(codeSearchPrompt, llmKey);

    expect(results).toHaveLength(1);
    expect(results).toEqual(
      expect.arrayContaining([expect.objectContaining(initialGeneration)])
    );
    expect(codeResults).toHaveLength(1);
    expect(codeResults).toEqual(
      expect.arrayContaining([expect.objectContaining(initialCodeGeneration)])
    );
  });

  test("can clear cache", async () => {
    await Promise.all([
      semanticCache.update("prompt test 1", llmKey, [
        { text: "generation test 1" },
      ]),
      semanticCache.update("prompt test 1", llmKey, [
        { text: "generation test 1" },
      ]),
      semanticCache.update("prompt test 1", llmKey, [
        { text: "generation test 1" },
      ]),
    ]);

    await semanticCache.clear(llmKey);

    const results = await semanticCache.lookup("prompt test 1", llmKey);

    expect(results).toBeNull();
  });
});

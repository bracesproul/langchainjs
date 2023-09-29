import { similarity as ml_distance_similarity } from "ml-distance";
import crypto from "crypto";
import { VectorStore } from "./base.js";
import { Embeddings } from "../embeddings/base.js";
import { Document } from "../document.js";
import { ScoreThresholdRetriever } from "../retrievers/score_threshold.js";
import { Generation, GenerationChunk } from "../schema/index.js";

/**
 * Interface representing a vector in memory. It includes the content
 * (text), the corresponding embedding (vector), and any associated
 * metadata.
 */
interface MemoryVector {
  content: string;
  embedding: number[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  metadata: Record<string, any>;
}

/**
 * Interface for the arguments that can be passed to the
 * `MemoryVectorStore` constructor. It includes an optional `similarity`
 * function.
 */
export interface MemoryVectorStoreArgs {
  similarity?: typeof ml_distance_similarity.cosine;
}

/**
 * Class that extends `VectorStore` to store vectors in memory. Provides
 * methods for adding documents, performing similarity searches, and
 * creating instances from texts, documents, or an existing index.
 */
export class MemoryVectorStore extends VectorStore {
  declare FilterType: (doc: Document) => boolean;

  memoryVectors: MemoryVector[] = [];

  similarity: typeof ml_distance_similarity.cosine;

  _vectorstoreType(): string {
    return "memory";
  }

  constructor(
    embeddings: Embeddings,
    { similarity, ...rest }: MemoryVectorStoreArgs = {}
  ) {
    super(embeddings, rest);

    this.similarity = similarity ?? ml_distance_similarity.cosine;
  }

  /**
   * Method to add documents to the memory vector store. It extracts the
   * text from each document, generates embeddings for them, and adds the
   * resulting vectors to the store.
   * @param documents Array of `Document` instances to be added to the store.
   * @returns Promise that resolves when all documents have been added.
   */
  async addDocuments(documents: Document[]): Promise<void> {
    const texts = documents.map(({ pageContent }) => pageContent);
    return this.addVectors(
      await this.embeddings.embedDocuments(texts),
      documents
    );
  }

  /**
   * Method to add vectors to the memory vector store. It creates
   * `MemoryVector` instances for each vector and document pair and adds
   * them to the store.
   * @param vectors Array of vectors to be added to the store.
   * @param documents Array of `Document` instances corresponding to the vectors.
   * @returns Promise that resolves when all vectors have been added.
   */
  async addVectors(vectors: number[][], documents: Document[]): Promise<void> {
    const memoryVectors = vectors.map((embedding, idx) => ({
      content: documents[idx].pageContent,
      embedding,
      metadata: documents[idx].metadata,
    }));

    this.memoryVectors = this.memoryVectors.concat(memoryVectors);
  }

  /**
   * Method to perform a similarity search in the memory vector store. It
   * calculates the similarity between the query vector and each vector in
   * the store, sorts the results by similarity, and returns the top `k`
   * results along with their scores.
   * @param query Query vector to compare against the vectors in the store.
   * @param k Number of top results to return.
   * @param filter Optional filter function to apply to the vectors before performing the search.
   * @returns Promise that resolves with an array of tuples, each containing a `Document` and its similarity score.
   */
  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: this["FilterType"]
  ): Promise<[Document, number][]> {
    const filterFunction = (memoryVector: MemoryVector) => {
      if (!filter) {
        return true;
      }

      const doc = new Document({
        metadata: memoryVector.metadata,
        pageContent: memoryVector.content,
      });
      return filter(doc);
    };
    const filteredMemoryVectors = this.memoryVectors.filter(filterFunction);
    const searches = filteredMemoryVectors
      .map((vector, index) => ({
        similarity: this.similarity(query, vector.embedding),
        index,
      }))
      .sort((a, b) => (a.similarity > b.similarity ? -1 : 0))
      .slice(0, k);

    const result: [Document, number][] = searches.map((search) => [
      new Document({
        metadata: filteredMemoryVectors[search.index].metadata,
        pageContent: filteredMemoryVectors[search.index].content,
      }),
      search.similarity,
    ]);

    return result;
  }

  /**
   * Static method to create a `MemoryVectorStore` instance from an array of
   * texts. It creates a `Document` for each text and metadata pair, and
   * adds them to the store.
   * @param texts Array of texts to be added to the store.
   * @param metadatas Array or single object of metadata corresponding to the texts.
   * @param embeddings `Embeddings` instance used to generate embeddings for the texts.
   * @param dbConfig Optional `MemoryVectorStoreArgs` to configure the `MemoryVectorStore` instance.
   * @returns Promise that resolves with a new `MemoryVectorStore` instance.
   */
  static async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: Embeddings,
    dbConfig?: MemoryVectorStoreArgs
  ): Promise<MemoryVectorStore> {
    const docs: Document[] = [];
    for (let i = 0; i < texts.length; i += 1) {
      const metadata = Array.isArray(metadatas) ? metadatas[i] : metadatas;
      const newDoc = new Document({
        pageContent: texts[i],
        metadata,
      });
      docs.push(newDoc);
    }
    return MemoryVectorStore.fromDocuments(docs, embeddings, dbConfig);
  }

  /**
   * Static method to create a `MemoryVectorStore` instance from an array of
   * `Document` instances. It adds the documents to the store.
   * @param docs Array of `Document` instances to be added to the store.
   * @param embeddings `Embeddings` instance used to generate embeddings for the documents.
   * @param dbConfig Optional `MemoryVectorStoreArgs` to configure the `MemoryVectorStore` instance.
   * @returns Promise that resolves with a new `MemoryVectorStore` instance.
   */
  static async fromDocuments(
    docs: Document[],
    embeddings: Embeddings,
    dbConfig?: MemoryVectorStoreArgs
  ): Promise<MemoryVectorStore> {
    const instance = new this(embeddings, dbConfig);
    await instance.addDocuments(docs);
    return instance;
  }

  /**
   * Static method to create a `MemoryVectorStore` instance from an existing
   * index. It creates a new `MemoryVectorStore` instance without adding any
   * documents or vectors.
   * @param embeddings `Embeddings` instance used to generate embeddings for the documents.
   * @param dbConfig Optional `MemoryVectorStoreArgs` to configure the `MemoryVectorStore` instance.
   * @returns Promise that resolves with a new `MemoryVectorStore` instance.
   */
  static async fromExistingIndex(
    embeddings: Embeddings,
    dbConfig?: MemoryVectorStoreArgs
  ): Promise<MemoryVectorStore> {
    const instance = new this(embeddings, dbConfig);
    return instance;
  }

  async dropIndex() {
    this.memoryVectors = [];
  }
}

interface CacheDict {
  [index: string]: MemoryVectorStore;
}

function hash(input: string) {
  return crypto.createHash("md5").update(input, "utf8").digest("hex");
}

export interface MemorySemanticVectorStoreArgs extends MemoryVectorStoreArgs {
  /**
   * The threshold for the similarity score results.
   * @default 0.2
   */
  scoreThreshold?: number;
  /**
   * The embeddings to use for the semantic cache.
   */
  embeddings: Embeddings;
}

/**
 * Load generations from json.
 *
 * @param {string} generationsJson - A string of json representing a list of generations.
 * @throws {Error} Could not decode json string to list of generations.
 * @returns {Generation[]} A list of generations.
 */
function loadGenerationsFromJson(generationsJson: string): GenerationChunk[] {
  try {
    const results = JSON.parse(generationsJson);
    return results.map(
      (generationDict: {
        text: string;
        generationInfo?: Record<string, any>;
      }) => {
        if (typeof generationDict.text !== "string") {
          throw new Error(`Invalid generation text: ${generationDict.text}`);
        }
        return new GenerationChunk(generationDict);
      }
    );
  } catch (error) {
    throw new Error(
      `Could not decode json to list of generations: ${generationsJson}`
    );
  }
}

function dumpGenerationsToJson(generations: Generation[]): string {
  return JSON.stringify(
    generations.map((generation) => ({ text: generation.text }))
  );
}

export class MemorySemanticVectorStore extends MemoryVectorStore {
  private cacheDict: CacheDict;

  private scoreThreshold: number;

  constructor({
    embeddings,
    scoreThreshold = 0.2,
    ...rest
  }: MemorySemanticVectorStoreArgs) {
    super(embeddings, rest);

    this.cacheDict = {};
    this.scoreThreshold = scoreThreshold;
  }

  private indexName(llmKey: string): string {
    const hashedIndex = hash(llmKey);
    return `cache:${hashedIndex}`;
  }

  private async getLlmCache(
    llmKey: string
  ): Promise<ScoreThresholdRetriever<MemoryVectorStore>> {
    const indexName = this.indexName(llmKey);

    const scoreThresholdOptions = {
      minSimilarityScore: this.scoreThreshold,
      maxK: 1,
    };

    if (indexName in this.cacheDict) {
      return ScoreThresholdRetriever.fromVectorStore(
        this.cacheDict[indexName],
        scoreThresholdOptions
      );
    }

    this.cacheDict[indexName] = await MemoryVectorStore.fromExistingIndex(
      this.embeddings
    );

    return ScoreThresholdRetriever.fromVectorStore(
      this.cacheDict[indexName],
      scoreThresholdOptions
    );
  }

  async clear(llmKey: string): Promise<void> {
    const indexName = this.indexName(llmKey);

    if (indexName in this.cacheDict) {
      await this.cacheDict[indexName].dropIndex();
      delete this.cacheDict[indexName];
    }
  }

  async lookup(prompt: string, llmKey: string): Promise<Generation[] | null> {
    const llmCache = await this.getLlmCache(llmKey);
    const results = await llmCache.getRelevantDocuments(prompt);

    let generations: Generation[] = [];
    if (results) {
      for (const document of results) {
        generations = generations.concat(
          loadGenerationsFromJson(document.metadata.return_val)
        );
      }
    }

    return generations.length > 0 ? generations : null;
  }

  async update(
    prompt: string,
    llmKey: string,
    returnVal: Generation[]
  ): Promise<void> {
    const llmCache = await this.getLlmCache(llmKey);

    const metadata = {
      llm_string: llmKey,
      prompt,
      return_val: dumpGenerationsToJson(returnVal),
    };
    const document = new Document({
      pageContent: prompt,
      metadata,
    });

    await llmCache.addDocuments([document]);
  }
}

import OpenAI from 'openai';
import { Embedding, EmbeddingVector } from './base-embedding';
import { envManager } from '../utils/env-manager';

export interface OpenRouterEmbeddingConfig {
    model: string;
    apiKey: string;
    baseURL?: string;
}

const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';

export class OpenRouterEmbedding extends Embedding {
    private client: OpenAI;
    private config: OpenRouterEmbeddingConfig;
    private dimension: number = 1536;
    protected maxTokens: number = 8192;

    constructor(config: OpenRouterEmbeddingConfig) {
        super();
        this.config = config;
        this.client = new OpenAI({
            apiKey: config.apiKey,
            baseURL: config.baseURL || OPENROUTER_BASE_URL,
        });

        const model = config.model || 'openai/text-embedding-3-small';
        const knownModels = OpenRouterEmbedding.getSupportedModels();
        if (knownModels[model]) {
            this.dimension = knownModels[model].dimension;
            this.maxTokens = knownModels[model].maxTokens || 8192;
        }
    }

    async detectDimension(testText: string = "test"): Promise<number> {
        const model = this.config.model || 'openai/text-embedding-3-small';
        const knownModels = OpenRouterEmbedding.getSupportedModels();

        if (knownModels[model]) {
            return knownModels[model].dimension;
        }

        try {
            const processedText = this.preprocessText(testText);
            const response = await this.client.embeddings.create(
                this.buildCreateParams(model, processedText) as any
            );
            return response.data[0].embedding.length;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            throw new Error(`Failed to detect dimension for model ${model}: ${errorMessage}`);
        }
    }

    async embed(text: string): Promise<EmbeddingVector> {
        const processedText = this.preprocessText(text);
        const model = this.config.model || 'openai/text-embedding-3-small';

        try {
            const response = await this.client.embeddings.create(
                this.buildCreateParams(model, processedText) as any
            );

            this.dimension = response.data[0].embedding.length;

            return {
                vector: response.data[0].embedding,
                dimension: this.dimension
            };
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            throw new Error(`Failed to generate OpenRouter embedding: ${errorMessage}`);
        }
    }

    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
        const processedTexts = this.preprocessTexts(texts);
        const model = this.config.model || 'openai/text-embedding-3-small';

        try {
            const response = await this.client.embeddings.create(
                this.buildCreateParams(model, processedTexts) as any
            );

            this.dimension = response.data[0].embedding.length;

            return response.data.map((item) => ({
                vector: item.embedding,
                dimension: this.dimension
            }));
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            throw new Error(`Failed to generate OpenRouter batch embeddings: ${errorMessage}`);
        }
    }

    getDimension(): number {
        const model = this.config.model || 'openai/text-embedding-3-small';
        const knownModels = OpenRouterEmbedding.getSupportedModels();

        if (knownModels[model]) {
            return knownModels[model].dimension;
        }

        return this.dimension;
    }

    getProvider(): string {
        return 'OpenRouter';
    }

    async setModel(model: string): Promise<void> {
        this.config.model = model;
        const knownModels = OpenRouterEmbedding.getSupportedModels();
        if (knownModels[model]) {
            this.dimension = knownModels[model].dimension;
            this.maxTokens = knownModels[model].maxTokens || 8192;
        } else {
            this.dimension = await this.detectDimension();
        }
    }

    getClient(): OpenAI {
        return this.client;
    }

    static getSupportedModels(): Record<string, { dimension: number; maxTokens?: number; description: string }> {
        return {
            'openai/text-embedding-3-small': {
                dimension: 1536,
                maxTokens: 8192,
                description: 'OpenAI text-embedding-3-small via OpenRouter'
            },
            'openai/text-embedding-3-large': {
                dimension: 3072,
                maxTokens: 8192,
                description: 'OpenAI text-embedding-3-large via OpenRouter'
            },
            'openai/text-embedding-ada-002': {
                dimension: 1536,
                maxTokens: 8192,
                description: 'OpenAI text-embedding-ada-002 via OpenRouter'
            },
            'qwen/qwen3-embedding-4b': {
                dimension: 2560,
                maxTokens: 32768,
                description: 'Qwen3-Embedding-4B (code-tuned, MTEB-Code top-tier) via OpenRouter'
            },
            'qwen/qwen3-embedding-8b': {
                dimension: 4096,
                maxTokens: 32768,
                description: 'Qwen3-Embedding-8B (MTEB-Code 80.7, cheapest provider ~$0.01/M) via OpenRouter'
            },
        };
    }

    /**
     * OpenRouter provider routing (sent in the request body). Controls which upstream provider
     * serves the embed request — used here to route to the CHEAPEST provider and skip pricey ones.
     * Env:
     *   OPENROUTER_PROVIDER_SORT     default "price" (cheapest first); also "throughput"|"latency"
     *   OPENROUTER_PROVIDER_ONLY     comma list, e.g. "nebius,deepinfra" (allow only these)
     *   OPENROUTER_PROVIDER_IGNORE   comma list, e.g. "siliconflow" (exclude these)
     *   OPENROUTER_MAX_PROMPT_PRICE  number, e.g. "0.01" (hard cap $/M prompt tokens)
     */
    private providerRouting(): Record<string, unknown> | undefined {
        const prefs: Record<string, unknown> = {};
        prefs.sort = envManager.get('OPENROUTER_PROVIDER_SORT') || 'price';
        const only = envManager.get('OPENROUTER_PROVIDER_ONLY');
        if (only) prefs.only = only.split(',').map(s => s.trim()).filter(Boolean);
        const ignore = envManager.get('OPENROUTER_PROVIDER_IGNORE');
        if (ignore) prefs.ignore = ignore.split(',').map(s => s.trim()).filter(Boolean);
        const maxPrompt = envManager.get('OPENROUTER_MAX_PROMPT_PRICE');
        if (maxPrompt) prefs.max_price = { prompt: parseFloat(maxPrompt) };
        return Object.keys(prefs).length ? prefs : undefined;
    }

    private buildCreateParams(model: string, input: string | string[]): Record<string, unknown> {
        const params: Record<string, unknown> = { model, input, encoding_format: 'float' };
        const provider = this.providerRouting();
        if (provider) params.provider = provider;
        return params;
    }
}

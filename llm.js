import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from "@google/generative-ai";
import debug from "debug";
import 'dotenv/config';

const llmDebug = new debug('llm');
const embeddingDebug = new debug('embedding');

// Initialize the Google Generative AI with your API key
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);

// Generate a vector embedding from the submitted prompt
export async function generateEmbedding(prompt) {
  const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" });
  llmDebug(`Model: embedding-001`);
  llmDebug(`Prompt: ${prompt}`);

  // This is the call to the Google Generative AI model that generates the embedding
  const result = await embeddingModel.embedContent(prompt);
  const embedding = result.embedding.values;

  embeddingDebug('Embedding:');
  embeddingDebug(embedding);

  return embedding;
}

// Use the Gemini model to generate an LLM response
// based on the prompt and any additional context obtained via RAG
export async function generateChatbotResponse(prompt, context) {
  const completionsModel = "gemini-1.5-pro";

  // The developer prompt to the chatbot tells it how to behave,
  // provides information about the data it's receiving,
  // and supplies the RAG context (if any)
  const developerPrompt = `
    You are a very enthusiastic HR representative who loves
    to help people!

    Given the following sections from the company 
    handbook and internal documents, answer the question using
    that information as the primary source.
    
    You can supplement the information in the context sections
    with general information that you know, but be sure to distinguish
    internal information from external information in your response.

    If you are unsure and the answer is not explicitly written
    in the information you have, say
    "Sorry, I don't know how to help with that."

    Context sections:
    ${context}

    Answer in conversational prose.
`;

  llmDebug(`Model: ${completionsModel}`);
  llmDebug(`Prompt: ${prompt}`);

  // Configure the Gemini model
  const geminiModel = genAI.getGenerativeModel({
    model: completionsModel,
    safetySettings: [
      {
        category: HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      },
      {
        category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      },
      {
        category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      },
      {
        category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      },
    ],
  });

  // Start a chat session
  const chat = geminiModel.startChat({
    history: [
      { role: "user", parts: [{ text: developerPrompt }] },
    ],
  });

  // Generate a response
  const result = await chat.sendMessage(prompt);
  const response = result.response;

  llmDebug("Response:");
  llmDebug(response);
  
  return `${response.text()}\n`;
}

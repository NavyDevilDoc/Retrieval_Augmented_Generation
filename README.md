# Retrieval_Augmented_Generation
Basic RAG model using open and closed source LLM and vector database

This has been a long time coming and basic as it may be, I'm pretty damn proud of it. Thanks to a never-ending stream of tutorials, documentation, and beating my head into the desk, I built a retrieval-augmented generation model that uses a mixture of open-source and OpenAI backbone LLMs. 

From Ollama, I downloaded Llama2-7B, Mixtral 8x7B, and Gemma. Llama2 and Gemma took an average of 8 m 34 s to generate an answer, and Mixtral took...forever and a day to the point where I killed the application. From OpenAI, I used GPT-3.5 Turbo, which took an average of 6.1 s to generate an answer. Mind you, I'm running a Mk 1 Mod 0 CPU, so a more powerful computer would likely do MUCH better. Frankly, I might upgrade my hardware to something that doesn't take forever. 

The vector database I use is Pinecone. It's a damn good system that's fairly intuitive, giving the option to build your database on the website or in the code. Originally, I built mine on the website, but the Pinecone documentation made it evident that creating a storage space in the code wasn't impossible for a guy like me. Langchain is the glue that holds it all together; thankfully, it's free. They do some amazing work over there!

Please note that OpenAI charges to use GPT-3.5 and the embedding tool and Pinecone charges a relatively small amount for their services. Using an open-source LLM gives you access to the Ollama embedding functionality, which is similarly free. I have notes in the code about the vector length for whichever LLM model you'd like to use with your database of choice. If you use OpenAI and/or Pinecone, you'll need to generate your API key, because you can't use mine. 

My next big update will be a simple Flask-based web application with minor tweaks and improvements to the base RAG model.

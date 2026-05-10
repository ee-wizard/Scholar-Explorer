# MIRACL-VISION: A Large, multilingual, visual document retrieval benchmark

Radek Osmulski\* NVIDIA Brisbane, Australia rosmulski@nvidia.com

Gabriel de Souza P. Moreira\* NVIDIA São Paulo, Brazil gmoreira@nvidia.com

Ronay Ak\*   
NVIDIA   
Sarasota, USA   
ronaya@nvidia.com   
Mengyao Xu\*   
NVIDIA   
Santa Clara, USA   
mengyaox@nvidia.com   
Benedikt Schifferer∗   
NVIDIA   
Berlin, Germany   
bschifferer@nvidia.com

Even Oldridge NVIDIA Vancouver, Canada eoldridge@nvidia.com

# Abstract

Document retrieval is an important task for search and Retrieval-Augmented Generation (RAG) applications. Large Language Models (LLMs) have contributed to improving the accuracy of text-based document retrieval. However, documents with complex layout and visual elements like tables, charts and infographics are not perfectly represented in textual format. Recently, image-based document retrieval pipelines have become popular, which use visual large language models (VLMs) to retrieve relevant page images given a query. Current evaluation benchmarks on visual document retrieval are limited, as they primarily focus only English language, rely on synthetically generated questions and offer a small corpus size. Therefore, we introduce MIRACL-VISION1, a multilingual visual document retrieval evaluation benchmark. MIRACL-VISION covers 18 languages, and is an extension of the MIRACL dataset, a popular benchmark to evaluate text-based multilingual retrieval pipelines. MIRACL was built using a human-intensive annotation process to generate high-quality questions. In order to reduce MIRACL-VISION corpus size to make evaluation more compute friendly while keeping the datasets challenging, we have designed a method for eliminating the "easy" negatives from the corpus. We conducted extensive experiments comparing MIRACL-VISION with other benchmarks, using popular public text and image models. We observe a gap in state-of-the-art VLM-based embedding models on multilingual capabilities, with up to $5 9 . 7 \%$ lower retrieval accuracy than a text-based retrieval models. Even for the English language, the visual models retrieval accuracy is $1 2 . 1 \%$ lower compared to textbased models. MIRACL-VISION is a challenging, representative, multilingual evaluation benchmark for visual retrieval pipelines and will help the community build robust models for document retrieval.

# Keywords

multilingual retrieval dataset, document retrieval, VLM, page retrieval, text retrieval, benchmark.

# 1 Introduction

Retrieval-Augmented Generation (RAG) has become a popular approach to provide context for Large Language Models, enabling

![](images/40bfeb1387c8b27c00e69a744c8e77ce2ab8d5c8ba82fea3f88d37bb7d755ca9.jpg)  
Figure 1: Example of a User Query and Document Image of MIRACL Vision

LLMs to answer zero-shot questions, e.g., about content that was not seen during training.

Many companies have been adopting RAG to create assistants that leverage their internal documents - like reports, contracts, presentations - to improve their customer service or increase productivity and quality of their internal processes.

A key component of RAG applications is retrieval. In a typical text-based retrieval pipeline, documents need to be first parsed for text extraction, which is split into chunks that are embedded for dense retrieval. Older scanned documents are represented as images and require Optical Character Recognition (OCR) to extract text. Most modern document formats store the actual text and avoid the need of OCR, but more complex document layouts (e.g.

![](images/510542f59f90e008d6b15e68f4503986a2945e1829417176e932e9b9c1d6bff5.jpg)  
Figure 2: Visualization of a text-based and image-based RAG pipeline

two-column documents, text interleaved with images and tables) make text extraction more challenging. This text-based retrieval scenario demands non-trivial ingestion and indexing pipelines for documents, which might involve specialized models. For example, document layout detection models to segment the page elements, usage of LLMs to caption figures and tables in natural language, a chunking strategy that is aware of the structure of the document.

A recent approach has been to represent pages as images[9] and retrieve them using Visual LLMs (VLMs), which have built-in OCR capabilities. VLMs are generation models capable of taking both text and images as input.

VLMs have been adapted as multimodal constrastive embedding models, that can align images and text representation in a shared embedding space. Recent VLM-based document retrieval models have been released, such as DSE-Qwen2[9], GME-Qwen2[18], Col-Pali and ColQwen[4].

Some benchmarks have been introduced to evaluate VLM-based embedding models capability to retrieve document pages: ViDoRe[5] and VDR[8]. However, they have some limitations, like questions generated synthetically, small and not challenging document corpus, and lack of multi-language coverage.

In this paper, we introduce the MIRACL-VISION benchmark, which is focused on evaluating the multilingual support of visual document retrieval.

MIRACL-VISION is based on MIRACL, a popular benchmark for multilingual text-based retrieval, including high-resource (e.g. English) and low-resource languages (e.g. Swahili), and languages with non-Latin alphabets (e.g., Arabic, Japanese, Korean, Russian). MIR-ACL authors invested a significant effort to collect representative user questions from Wikipedia articles by native speakers.

In MIRACL-VISION data collection process, we leverage the highquality MIRACL questions about Wikipedia articles in multiple languages, generate the corresponding images of the first page of those articles, and filter the dataset to reduce its size while keeping hard-negatives / distractors for retrieval evaluation.

The major contributions of this paper are summarized as follows:

• The release of MIRACL-VISION, a comprehensive benchmark for evaluation of multilingual visual document retrieval and its comparison with existing benchmarks; • We describe the data collection process for MIRACL-VISION, which can be adapted to create visual retrieval versions of other text retrieval datasets based on documents or webpages; • We provide a benchmark of state-of-the-art visual document embedding models on multilingual retrieval task with MIRACL-VISION and compare them with text-based embedding models on an equivalent text-based dataset.

We believe MIRACL-VISION will be helpful for the community to evaluate the multilingual capabilities of vision-based retriever pipelines.

# 2 Background

In this section, we discuss related work on retrieval benchmarks and VLM-based constrastive embedding models.

# 2.1 Text retrieval benchmarks

Machine Learning benchmarks are important to help the community to set targets for real-world problems and tasks, gauge research progress towards those goals over time, and provide a common ground to compare different methods and models.

One of the main benchmarks for text information retrieval is BEIR[11]. It is a selection of 18 English retrieval datasets from 9 heterogeneous retrieval tasks, including Question Answering retrieval datasets - NQ, HotpotQA and FiQA-2018 - that are relevant for RAG applications.

MIRACL [16] is a multilingual benchmark for text retrieval, comprising 18 different languages that cover over three billion native speakers around the world. Their authors leveraged native speakers to generate around 77k queries and evaluate top-k querypassage pairs produced by a retrieval system. This careful humanannotation has been very valuable for multilingual text retrieval evaluation. Since there is no comprehensive multilingual benchmark for visual document retrieval, as we discuss in the next section, we introduce MIRACL-VISION in this work.

# 2.2 Visual document retrieval benchmarks

The Visual Document Retrieval Benchmark (ViDoRe)[5] is popular for evaluating page-level document retrieval. It covers many document types (e.g., financial reports, administrative and medical documents, academic papers, among others) and features questions about different visual elements (text, tables, charts, infographics). Vi-DoRe adapts 6 existing Visual Question Answering (VQA) datasets for retrieval, and create other 5 datasets by generating questions using a proprietary VLM from corpuses of documents. The majority of its datasets are in English, only two of them cover French language.

VDR-Multilingual2[8] is another benchmark for visual document retrieval. It covers English, French, German, Italian, and Spanish languages. For each language and category of questions (text, visual, mix) there are 100 questions and a corpus 1000 page images. A

VLM was used to generate synthetic questions that were humanreviewed.

We compare our MIRACL-VISION with ViDoRe and VDR-Multilingual benchmarks in Section 3.2.2

# 2.3 VLM-based embedding models

Contrastive dense embedding models represent variable-length information as a fixed dimension vector that can be used for downstream tasks, like retrieval. Transformer models have been finetuned to serve as text embedding models using encoder-based architectures (DPR[6], E5 [12]), and decoder models (E5-Mistral [13]). Multilingual text embedding models have been released, like multilingual-e5-large [14], snowflake-arctic-embed-l [15], bge-m3 [3], and gte-multilingual-base [17].

Visual LLM models (e.g. PaliGemma[10], SmolVL[1], QwenVL[2], and Eagle2[7]) combine vision and language capabilities, enabling tasks like image captioning, question answering, and multimodal retrieval.

VLMs typically integrate a vision encoder model (e.g. SigLIP) with a Language model (e.g. Llama) by using a connector (e.g. MLP) that projects and aligns the text and image embedding spaces.

VLMs can be adapted from a generation model into and multimodal embedding model by pooling the Transformer embedding outputs and training it with constrastive learning, bringing together in the embedding space the positive text-image pairs, e.g., matching a question with the corresponding document page image that contains the answer.

Some recent representative VLM-based models for visual document retrieval are dse-qwen2- $2 b$ -mrl-v1[9], gme-Qwen2-VL-2B-Instruct[18], vdr- $_ { \cdot 2 b }$ -multi-v1, and colqwen2-v1.0[4]. In Section 4.2, we evaluate those models on visual document retrieval benchmarks, including our MIRACL-VISION.

# 3 Methodology

In this section, we describe our design and process for extending MIRACL and generating the MIRACL-VISION data set to benchmark multilingual visual document retrieval (Section 3.1). Afterwards, we compare the characteristics of MIRACL-VISION with other datasets in Section 3.2.

# 3.1 MIRACL-VISION generation process

We illustrate our general process to extend MIRACL to MIRACL VISION in Figure 3, which is inspired by the construction of the Wiki-SS dataset [9]

To generate MIRACL-VISION, we have designed a process that reuses MIRACL human-generated questions and replaces the Ground Truth (GT) annotated text passages, i.e., that contain the answer, by an image of the document where the GT passage is contained. That process also involves steps to reduce dataset size while keeping hard-negatives for all questions. We detail the steps in this section.

# Step 1. Filtering 1st Paragraph per Article

MIRACL corpus was extracted from Wikipedia articles, that can be long. For that reason, they are split into multiple chunks to keep their length manageable to be embedded. For example, English Wikipedia has 5.7M articles and the corpus of English MIRACL has 32M chunks.

To be able to reuse MIRACL questions for visual retrieval, we had to design a process to locate the chunk containing the answer within an article and extract the corresponding document page image containing that chunk. We did not find a reliable solution to extract images from chunks in any part of the document. We simplified the process by keeping only the first chunk. That way, we can always take the first page of a Wikipedia article, as it is ensured to contain the first paragraph.

# Step 2. Selecting Answerable Questions

After we removed all chunks which are not the first paragraph, some questions did not have the corresponding GT anymore in the corpus. In this step, we removed all the questions that do not have any positive document in the corpus. We name this intermediate dataset as MIRACL-1stParagraph.

# Step 3. Reducing Corpus Size while keeping it challenging for evaluation

As described in Table 1, some languages still have a large corpus after keeping only the first paragraph chunks, such as English with 5.7M documents / chunks. For a large number of documents, extracting document images and running the evaluation pipeline would be costly and require significant computational resources, making it impractical as an evaluation dataset for the research community. Besides that, most documents of a large corpus are irrelevant to annotated queries, as it is easy to distinguish them from the correct documents.

We have designed a method to reduce the dataset size, while keeping its hardness for retrieval evaluation. It only keeps documents in the corpus that are either positives or hard negatives for at least one question. To perform that filtering, we use the multilingual-e5-large text embedding model to embed all questions and documents, compute the cosine similarity among those embeddings to get the top-k (top-100 for English and top-50 for other languages) most similar documents to the question and only keep them in the corpus. The resulting corpus is much smaller, but still challenging for retrieval, as it keeps the main distractor documents for each query. We name this intermediate dataset as MIRACL-1stParagraph-Reduced.

# Step 4. Generating Image and Text

MIRACL is based on Wikipedia, a publicly available website with user-friendly terms and conditions. We follow a similar process as described in [9]. For each document in MIRACL, we download the corresponding Wikipedia article. We modify the HTML code to render only relevant content, removing some elements like sidebar, and header. Then we extract an image of the first vertical 2048 pixels of the article with Playwright3, crop it to $9 8 0 \mathrm { p x } \mathrm { x } 9 8 0 \mathrm { p x }$ pixels and save it to disk. You can see examples in Figure 1 and in Appendix A of user queries with corresponding Wikipedia article containing the answer in the first paragraph.

In addition, we extract the text from the HTML body, keeping the first 12 sentences4 as an approximate text representation of the extracted image. We name this dataset as MIRACL-VISION-text.

Appendix A provides an example of extracted image and corresponding text from a Wikipedia article.

![](images/c78f30f80b4cb351b5d2ffcdef002e35336df529737cefa97a8f0c36a790e0f8.jpg)  
Figure 3: Visualization of the process to create MIRACL-VISION and the intermediate datasets

# 3.2 Comparison of retrieval benchmark datasets

In this section, we can compare statistics of MIRACL-VISION with other evaluation datasets.

3.2.1 Statistics of MIRACL and MIRACL-VISION. Table 1 provides the statistics of the MIRACL, MIRACL-1stParagraph and MIRACL-VISION. The original MIRACL has an average 5.9M text chunks per language and filtering on 1st paragraph reduces the corpus size to an average of 1M chunks. As some queries are not answerable with the filtered dataset, the average number of queries per language is reduced from 750 to 439.

A corpus of 1M images per language would require significant computation for evaluating models. Therefore, in Step 3 in Section 3.1 we describe how we reduce the corpus size to an average of 18,819 documents by removing chunks that are not relevant to any question, i.e. we only keep hard-negatives / distractors. This approach ensures correlated retrieval results while reducing corpus size and speeding up evaluation.

3.2.2 Comparison of visual document retrieval benchmarks. We compare the characteristics of MIRACL-VISION with other popular vision benchmarks in Table 2. ViDoRe provides 8 English and 2 French datasets; vdr-multilingual contains English, French, German, Italian, and Spanish datasets and MIRACL-VISION has a total of 18 different languages, including low-resource languages or languages with non-Latin alphabets. The number of queries per language dataset are for those datasets (300-483). The average corpus size per language of MIRACL-VISION is 6x larger than the other datasets.

One limitation of MIRACL-VISION is that the queries are mainly based on information from the text, whereas ViDoRe and vdrmultilingual datasets contain queries for tables, charts and infographics. However, we believe that text-based queries are highly relevant to evaluate the multilingual capabilities for vision-based models.

The researchers that prepared MIRACL dataset have trained native speakers to ask relevant questions given a "prompt" passage, where the prompt passage cannot answer the question but the question will likely be answered by the remaining text and verified it later.

ViDoRe and Vdr-multilingual generated synthetic queries for some datasets with LLMs and manually reviewed them. In our experience, prompting an LLM to formulate questions given a document has the tendency that specific keywords will be repeated in the questions. Therefore, the questions might not be representative of open queries from a user who is not biased by a specific document.

ViDoRe repurposed existing Visual Question-Answering (VQA) datasets as retrieval datasets. One particular issue of that approach is that they may contain queries that are specific to a sentence, table or image, and might not make sense for retrieval. For example, ViDORe’s docvqa test set contains question like "What is the table number?" or What is the email address provided?, which makes sense to ask a VLM when the image is provided, but doesn’t make sense for document retrieval.

# 4 Main results and discussion

We present in this section experiments results comparing MIRACL and derived datasets, MIRACL-VISION and other visual document retrieval datasets.

# 4.1 Retrieval accuracy on MIRACL and our derived text datasets

In this section, we compare multilingual text embedding models - dse-qwen2- $_ { 2 b }$ -mrl- $\cdot \nu I ^ { 5 }$ [9], gme-Qwen2-VL-2B-Instruct6[18], vdr-2bmulti- $\cdot \nu I ^ { 7 }$ , and colqwen2-v1 $\cdot  { \boldsymbol { O } } ^ { 8 } [ 4 ]$ - on original MIRACL and our intermediate MIRACL text variants that are compatible with MIRACL-VISION: MIRACL-1stParagraph-Reduced and MIRACL-VISIONtext, described in Section 3.1. We calculated all scores for every model and dataset ourselves.

The selected multilingual embedding models were trained on the original MIRACL train split. For a fair comparison with the vision models, we fine-tune Llama 3.2 1B as an embedding model with constrastive loss and some data excluding MIRACL train set as a baseline model. Its retrieval accuracy is not much smaller than the other multilingual text embedding models though.

As can be seen in Table 3, the average NDCG@10 over all models is 0.6499 for the original MIRACL dataset and 0.8271 for our MIRACL-1stParagraph. It indicates that our MIRACL-1stParagraph-Reduced filtered version is easier than MIRACL. One hypothesis is that questions related to the first paragraph are easier or that other chunks from the same Wikipedia article (which we remove) are more challenging negatives.

Table 1: Comparison of number of queries and number of documents between original MIRACL, MIRACL filtered on 1st paragraph and MIRACL-VISION.   

<table><tr><td rowspan="2">Language</td><td colspan="2">MIRACL (original)</td><td colspan="2">MIRACL-1stParagraph</td><td colspan="2">MIRACL-VISION</td></tr><tr><td># of queries</td><td># of document chunks</td><td># of queries</td><td># of documents</td><td># of queries</td><td># of documents</td></tr><tr><td>Arabic</td><td>2896</td><td>2061414</td><td>2127</td><td>656982</td><td>2127</td><td>75444</td></tr><tr><td>Bengali</td><td>411</td><td>297265</td><td>229</td><td>63762</td><td>229</td><td>8495</td></tr><tr><td>Chinese</td><td>393</td><td>4934368</td><td>189</td><td>1246389</td><td>189</td><td>8672</td></tr><tr><td>English</td><td>799</td><td>32893221</td><td>447</td><td>5758285</td><td>447</td><td>42971</td></tr><tr><td>Farsi</td><td>632</td><td>2207172</td><td>342</td><td>857827</td><td>342</td><td>15846</td></tr><tr><td>Finnish</td><td>1271</td><td>1883509</td><td>791</td><td>447815</td><td>791</td><td>33679</td></tr><tr><td>French</td><td>343</td><td>14636953</td><td>142</td><td>2325608</td><td>142</td><td>6990</td></tr><tr><td>German</td><td>305</td><td>15866222</td><td>129</td><td>2651352</td><td>129</td><td>6302</td></tr><tr><td>Hindi</td><td>350</td><td>506264</td><td>184</td><td>148107</td><td>184</td><td>8004</td></tr><tr><td>Indonesian</td><td>960</td><td>1446315</td><td>603</td><td>446330</td><td>603</td><td>23842</td></tr><tr><td>Japanese</td><td>860</td><td>6953614</td><td>387</td><td>1133444</td><td>387</td><td>17909</td></tr><tr><td>Korean</td><td>213</td><td>1486752</td><td>130</td><td>437373</td><td>130</td><td>5700</td></tr><tr><td>Russian</td><td>1252</td><td>9543918</td><td>564</td><td>1476045</td><td>564</td><td>25201</td></tr><tr><td>Spanish</td><td>648</td><td>10373953</td><td>369</td><td>1669181</td><td>369</td><td>17749</td></tr><tr><td>Swahili</td><td>482</td><td>131924</td><td>239</td><td>47793</td><td>239</td><td>7166</td></tr><tr><td>Telugu</td><td>828</td><td>518079</td><td>480</td><td>66353</td><td>480</td><td>15429</td></tr><tr><td>Thai</td><td>733</td><td>542166</td><td>451</td><td>128179</td><td>451</td><td>16313</td></tr><tr><td>Yoruba</td><td>119</td><td>49043</td><td>95</td><td>33094</td><td>95</td><td>3022</td></tr><tr><td>Average</td><td>750</td><td>5907342</td><td>439</td><td>1088551</td><td>439</td><td>18819</td></tr></table>

Table 2: Comparison of the characteristics of MIRACL-VISION with vidore and vdr-multilingual benchmarks.   

<table><tr><td></td><td>vidore</td><td>vdr-multilingual</td><td>MIRACL-VISION</td></tr><tr><td># of different languages</td><td>2</td><td>5</td><td>18</td></tr><tr><td># of datasets</td><td>10</td><td>5</td><td>18</td></tr><tr><td>avg. number of queries per dataset avg. number of documents per dataset</td><td>380 672</td><td>300</td><td>483</td></tr><tr><td>document selection</td><td>random</td><td>3000 random</td><td>18500 hard-negatives sampled from</td></tr><tr><td></td><td></td><td></td><td>large corpus</td></tr><tr><td>modalities</td><td>text, charts, tables, infographics.</td><td>text, visual.</td><td>text</td></tr><tr><td>query generation</td><td>human-generated and synthetic with manual evaluation</td><td>synthetic generated with man- ual evaluation</td><td>human-generated</td></tr></table>

Table $\mathbf { 3 } \mathbf { : N D C G } @ \mathbf { 1 0 }$ of text embedding models on MIRACL original and our variants   

<table><tr><td></td><td>MIRACL</td><td>MIRACL- MIRACL- 1stPara-</td><td>1stParagraph-VISION-</td><td>MIRACL-</td></tr><tr><td rowspan="5">Llama-3.2-1B (internal) gte-multilingual-base multilingual-e5-large arctic-embed-l-v2.0</td><td>0.6225</td><td>graph 0.8231</td><td>Reduced 0.8292</td><td>text 0.7932</td></tr><tr><td>0.6210</td><td>0.8072</td><td>0.8136</td><td>0.7682</td></tr><tr><td>0.6512</td><td>0.8322</td><td>0.8323</td><td>0.7624</td></tr><tr><td>0.6493</td><td>0.8289</td><td>0.8310</td><td>0.7806</td></tr><tr><td>0.6776</td><td>0.8442</td><td>0.8468</td><td>0.7964</td></tr><tr><td>bge-m3 Average</td><td>0.6499</td><td>0.8271</td><td>0.8306</td><td>0.7798</td></tr></table>

By comparing MIRACL-1stParagraph and MIRACL-1stParagraph-Reduced columns, we can notice they are close. That indicates that our method for reducing dataset size (by 58x) while keeping hard negatives for questions is successful in maintaining retrieval accuracy correlation.

We also evaluate the models on MIRACL-MIRACL-text, which is the MIRACL-VISION version with text extracted from HTML that roughly matches the textual content present in the 1st page of the Wikipedia article. The average score decreases by 3 percent points to 0.7798. As extracted HTML text is longer than the original MIRACL chunked text from 1st paragraph, we believe the additional noise might make it more challenging for the models to retrieve the right content. Overall, the MIRACL-VISION-text behaves similarly to the filtered MIRACL-1stParagraph data.

# 4.2 Retrieval accuracy of visual document retrieval datasets

We compare MIRACL-VISION to other visual document retrieval benchmarks - ViDoRe and vdr-multilingual - using 4 public VLMbased embedding models, as shown in Table 4. The model with best average NDCG@10 - colqwen2-v1.0 - scores 0.9604 for vdrmultilingual and 0.8969 for vidore benchmark. Both benchmarks are almost saturated. One hypothesis could be that their small corpus size, with less than 3000 documents, is not challenging for retrieval. Another possibility is that generating synthetic questions with VLMs or LLMs have the tendencies to repeat phrases and keywords in the queries, making it easier to retrieve the right chunks. These synthetic questions may differ from real user-generated open questions, which are not biased toward rephrasing fragments or key words of the documents they seek to retrieve. The visual retriever models have significantly lower $\mathrm { N D C G } @ 1 0$ in MIRACL-VISION (average 0.4715) than on other datasets, indicating it provides a challenging benchmark for multilingual visual document retrieval.

Table 4: NDCG@10 of VLM-based embedding models on visual document retrieval benchmarks   

<table><tr><td></td><td>vdr- multilingual</td><td>vidore</td><td>MIRACL-Vision</td></tr><tr><td>dse-qwen2-2b-mrl-v1</td><td>0.8363</td><td>0.8416</td><td>0.4426</td></tr><tr><td>gme-Qwen2-VL-2B-Inst</td><td>0.9165</td><td>0.8878</td><td>0.5283</td></tr><tr><td>vdr-2b-multi-v1</td><td>0.9371</td><td>0.8584</td><td>0.4741</td></tr><tr><td>colqwen2-v1.0</td><td>0.9604</td><td>0.8969</td><td>0.4728</td></tr><tr><td>Average</td><td>0.9126</td><td>0.8712</td><td>0.4795</td></tr></table>

# 4.3 How visual embedding models compare with text embedding models on multilingual document retrieval?

In Table 5, we compare the VLM-based embedding models on MIRACL-VISION with text embedding models on MIRACL-VISION text, as both datasets contain the same questions and documents (represented as a document screenshot or text). The best vision model is gme-Qwen2-VL-2B-Instruct with an average NDCG@10 score of 0.5283 and best public text-model is bge-m3 with 0.7964, outperforming the visual pipelines by over $5 0 \%$ , as seen in Table 5. A detailed analysis shows that the vision models do not work for Thelugu, with $\mathrm { N D C G } @ 1 0$ below $< 0 . 1$ . After removing the language as an outlier, the text-based models perform $4 3 \%$ higher in average compared to visual embedding models.

Table 5 provides the performance per language and we can see that the text-based versions perform better for every language. In case of English MIRACL, the gap is the smallest but still significant with $1 2 . 1 \%$ . Common languages, such as Chinese, French or Spanish, have a similar pattern with up to $1 6 . 5 \%$ . Arabic, Hindi and Thai, which are less common in research, wih non-Latin alphabet, have a performance gap of up to $5 9 . 7 \%$ .

Table 5 also shows the number of parameters per model. In the case of VLMs, it includes only the LLM without the vision backbone. The gte-multilingual-base outperforms the vision models with $_ { 5 \mathrm { x } }$ less parameters (305M parameters vs. 1543M parameters of Qwen2- based VLMs model).

As an additional note, vdr- $2 b$ -multi-v1 is a continuous fine-tuning of dse-qwen2-2b-mrl-v1 based on vdr-multilingual-train and reports significant gains on French and German on vdr-multilingual-test. Overall, vdr-2b-multi-v1 has a better performance than dse-qwen2- $2 b$ -mrl-v1 on MIRACL-VISION, but we do not observe similar gains, which might indicate that the Qwen2-based models have multilingual capabilities but additional fine-tuning learns the data distribution of vdr-multilingual-test.

# 5 Limitations

In this section, we discuss limitations and future research directions.

5.0.1 Questions only about text modality. As MIRACL is a textbased dataset, most user queries are answered by text paragraph and MIRACL-VISION’s modality is mainly text. The other visual document retrieval benchmark - ViDoRe and vdr-multilingual - provide questions backed by other modalities, such as charts, infographics or tables. However, we believe that the text modality is sufficient to evaluate multilingual capabilities of vision-based retrievers and our experiments demonstrate blindspots of current state-of-the-art models.

5.0.2 MIRACL-VISION-text could be refined to match perfectly MIRACL-VISION textual content. Most visual document retrieval pipelines assume PDFs as input. Generating an image per PDF page is easy, but extracting text can be more challenging. The comparison of MIRACL-VISION-text with vision-based MIRACL-VISION relies on text extraction from the HTML body of each article. The extracted text is clean and might have a higher quality as extracted text from PDF. One option is to use an OCR pipeline to convert PDFs to image to text, but as PDFs can contain the text as input, it can be a mixed solution. Comparing different OCR pipelines is beyond the scope of this paper and the extracted HTML text is an upper bound for high-quality PDF to text conversion.

# 6 Conclusion

In this paper, we introduced MIRACL-VISION, a multilingual benchmark for visual document retrieval. We described the methodology to generate MIRACL-VISION from MIRACL and Wikipedia articles, covering 18 different languages. We outlined our method for reducing the large corpus size by strategically selecting hard negatives. The resulting multilingual datasets are both challenging and efficiently sized, ensuring manageable computational requirements for evaluation.

Our experiments demonstrate that current state-of-the-art vision embedding models on text-heavy pages have a lower retrieval accuracy compared to smaller text embedding models, across all languages. The performance gap is up to $5 9 . 7 \%$ . Although some prior work suggests that the VLMs have zero-shot multilingual capabilities and that VLM-based document retrieval is superior to a text-based pipeline, MIRACL-VISION challenges the approach. We believe the release of MIRACL-VISION will enable the community to gauge their progress towards more robust multilingual vision embedding models.

Table 5: NDCG $@$ 10 of text embedding models and visual embedding models on MIRACL-VISI   

<table><tr><td rowspan="2"></td><td colspan="5">MIRACL-VISION (Text)</td><td colspan="4">MIRACL-VISION (Image)</td></tr><tr><td>multil- ingual- e5-large</td><td>arctic- embed- l-v2.0</td><td>gte- multilingual- base</td><td>bge-m3</td><td>Llama- 3.2-1B (inter-</td><td>dse- qwen2- 2b-mrl-</td><td>gme- Qwen2- VL-2B-</td><td>vdr-2b- multi- v1</td><td>colqwen2- v1.0</td></tr><tr><td># Params (in M)</td><td>560</td><td>567</td><td>305</td><td>567</td><td>nal) 1235</td><td>v1 1543</td><td>Instruct 1543</td><td>1543</td><td>1543</td></tr><tr><td>Language</td><td></td><td></td><td></td><td>0.8883</td><td></td><td></td><td>0.4888</td><td></td><td></td></tr><tr><td>Arabic</td><td>0.8557</td><td>0.8754 0.8325</td><td>0.8503</td><td>0.8585</td><td>0.8833 0.7902</td><td>0.3893 0.2352</td><td>0.3755</td><td>0.4379 0.2473</td><td>0.4129</td></tr><tr><td>Bengali</td><td>0.8421</td><td></td><td>0.8211</td><td></td><td></td><td>0.5962</td><td>0.6314</td><td></td><td>0.2888</td></tr><tr><td>Chinese</td><td>0.6900</td><td>0.7179</td><td>0.7167</td><td>0.7458</td><td>0.7561</td><td>0.6605</td><td></td><td>0.5963</td><td>0.4926</td></tr><tr><td>English</td><td>0.7029</td><td>0.7437</td><td>0.7345</td><td>0.7348</td><td>0.7721</td><td></td><td>0.6784 0.3085</td><td>0.6784</td><td>0.6417</td></tr><tr><td>Farsi</td><td>0.6793</td><td>0.7001</td><td>0.6984</td><td>0.7297</td><td>0.7192</td><td>0.2250 0.4162</td><td>0.6863</td><td>0.2398</td><td>0.2616</td></tr><tr><td>Finnish</td><td>0.8974</td><td>0.9014</td><td>0.8957</td><td>0.9071</td><td>0.9097</td><td>0.7160</td><td>0.6851</td><td>0.5283</td><td>0.6604</td></tr><tr><td>French</td><td>0.7208</td><td>0.8236</td><td>0.7771</td><td>0.8158</td><td>0.8545</td><td>0.6267</td><td>0.6345</td><td>0.7194</td><td>0.6876</td></tr><tr><td>German</td><td>0.7622</td><td>0.7774 0.7255</td><td>0.7498 0.6916</td><td>0.7695</td><td>0.7823 0.7770</td><td>0.1740</td><td>0.3127</td><td>0.6205</td><td>0.5995</td></tr><tr><td>Hindi</td><td>0.7595</td><td></td><td></td><td>0.7581</td><td>0.6977</td><td>0.4866</td><td>0.5416</td><td>0.2058</td><td>0.2209</td></tr><tr><td>Indonesian</td><td>0.6793 0.8378</td><td>0.6906 0.8484</td><td>0.6757 0.8442</td><td>0.7049 0.8720</td><td>0.8802</td><td>0.6232</td><td>0.7305</td><td>0.5254</td><td>0.5320</td></tr><tr><td>Japanese</td><td>0.7327</td><td>0.7545</td><td>0.7397</td><td>0.7934</td><td>0.8088</td><td>0.4446</td><td>0.6202</td><td>0.6553</td><td>0.6970</td></tr><tr><td>Korean Russian</td><td>0.7857</td><td>0.8242</td><td>0.8023</td><td>0.8363</td><td>0.8468</td><td>0.6505</td><td>0.7202</td><td>0.4952 0.6995</td><td>0.4419</td></tr><tr><td>Spanish</td><td>0.6596</td><td>0.7250</td><td>0.7029</td><td>0.7268</td><td>0.7318</td><td>0.5927</td><td>0.6277</td><td>0.6274</td><td>0.6811</td></tr><tr><td>Swahili</td><td>0.8157</td><td>0.8089</td><td>0.7987</td><td>0.8337</td><td>0.8059</td><td>0.4156</td><td>0.5348</td><td>0.4509</td><td>0.6224</td></tr><tr><td>Telugu</td><td>0.8948</td><td>0.9201</td><td>0.9076</td><td>0.9090</td><td>0.8101</td><td>0.0274</td><td>0.0893</td><td>0.0318</td><td>0.4931</td></tr><tr><td>Thai</td><td>0.8424</td><td>0.8485</td><td>0.8509</td><td>0.8682</td><td>0.8673</td><td>0.2692</td><td>0.3563</td><td>0.3177</td><td>0.0264</td></tr><tr><td>Yoruba</td><td>0.5655</td><td>0.5332</td><td>0.5698</td><td>0.5842</td><td>0.5839</td><td>0.4178</td><td>0.4884</td><td>0.4577</td><td>0.2389</td></tr><tr><td>Average</td><td>0.7624</td><td>0.7806</td><td>0.7682</td><td>0.7964</td><td>0.7932</td><td>0.4426</td><td>0.5283</td><td>0.4741</td><td>0.5120</td></tr><tr><td>Average w/o Thelugu</td><td>0.7546</td><td>0.7724</td><td>0.7600</td><td>0.7898</td><td>0.7922</td><td>0.4670</td><td>0.5542</td><td>0.5002</td><td>0.4728 0.4991</td></tr></table>

In the future, we plan to provide a MIRACL-VISION train split and fine-tune visual embedding models on it. We also suggest enriching MIRACL-VISION with more modalities in multiple languages for multimodal multilingual evaluation.
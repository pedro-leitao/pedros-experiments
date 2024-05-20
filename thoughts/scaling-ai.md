---
myst:
    html_meta:
        description: "There's been a lot of speculation on whether we're hitting the limits of our current approaches to scaling AI. Will OpenAI and others be able to continue to scale up models at the same pace? Will these new models be able to solve new problems, or will GPT-6 be just an iteration over GPT-4 ?"
        keywords: "AI, scaling, multimodal models, CLIP, Stable Diffusion, large language models, LLMs, zero-shot learning, data, data quality, data diversity, data problem, Chinchilla scaling law"
---

# On Scaling AI: Are we Hitting the Limits of our Current Approaches?

```{image} https://upload.wikimedia.org/wikipedia/commons/1/18/Chinchilla_lanigera_%28Wroclaw_zoo%29-2.JPG
:alt: The Chinchilla
:width: 320px
:align: right
```

There's been a lot of speculation on whether we're hitting the limits of our current approaches to scaling AI. Will OpenAI and others be able to continue to scale up models at the same pace? Will these new models be able to solve new problems, or will GPT-7 be just an iteration over GPT-4 ?

A recent paper regarding multi-modal models has clearly indicated that when it comes to CLIP and Stable-Diffusion, the scaling laws are not as favorable as one might think.

## The paper

The paper ["No 'Zero-Shot' Without Exponential Data"](https://arxiv.org/abs/2404.04125) examines how the performance of multimodal models like [CLIP](https://openai.com/index/clip/) and Stable Diffusion is influenced by the frequency of concepts in their pretraining datasets. It finds that these models require exponentially more data to achieve linear improvements in performance for "zero-shot" tasks, highlighting a significant sample inefficiency. This finding implies that the impressive performance of such models is largely due to extensive pretraining data rather than true zero-shot generalization capabilities.

```{index} Sample inefficiency
```

```{admonition} Sample inefficiency
:class: tip
"Sample inefficiency" refers to the need for an excessively large amount of data to achieve a relatively small improvement in performance. In the context of machine learning models, it indicates that as you scale the model or attempt to perform more complex tasks (like zero-shot learning), the amount of additional data required increases disproportionately compared to the performance gains. This inefficiency can make it costly and impractical to keep improving the model solely by increasing the dataset size, as the benefits diminish relative to the data volume and computational resources needed.
```

This research has significant implications for the scalability and future development of large language models (LLMs) like GPT-4. Much like multimodal models, LLMs also benefit from vast amounts of diverse training data to improve their performance on various tasks. The study's finding of a log-linear relationship between concept frequency and performance suggests that LLMs might also face similar challenges in achieving true zero-shot learning without exponentially increasing their training data.

## The implications

For instance, while LLMs can generate coherent and contextually appropriate text across various topics, their performance improves significantly when the training data includes a higher frequency of relevant concepts. This explains why models trained on extensive datasets, such as those encompassing diverse domains and large vocabularies, perform better in generating accurate and relevant responses across different subjects.

The implications for scaling LLMs are profound. To achieve robust zero-shot performance, simply increasing the size of the model may not be sufficient. Instead, it may require curating more extensive and diverse datasets, explicitly rebalancing concept distributions, and addressing issues such as data alignment and sample efficiency. As we push the boundaries of what these models can do, understanding the relationship between data frequency and model performance will be crucial for developing more efficient and capable AI systems.

This research underscores the importance of data quality and diversity in training large language models and multimodal models. It also highlights the challenges of achieving true zero-shot learning and generalization in AI systems.

## The data problem 

```{index} Data problem
```

Where will we get all the curated, diverse, and high-quality data needed to train these models effectively? The data problem is a significant bottleneck in scaling - there are limits to how much data is available and how quickly it can be processed. Additionally, the quality and diversity of data are crucial factors. Without addressing these issues, we may struggle to maintain the pace of progress seen in recent years.

Other [papers](https://arxiv.org/pdf/2305.13230) have discussed this, and the consensus seems to be that number of tokens trumps model size. This means that the amount of data used to train a model is more critical than the model's size in determining its performance. But, where will all the data come from ? Sure, there's still a lot of non-text data out there, but do we have the techniques to effectively use it in a way where models can learn from it ?

```{index} Chinchilla scaling law
```

```{admonition} The Chinchilla scaling law
:class: tip
The [Chinchilla scaling law](https://en.wikipedia.org/wiki/Neural_scaling_law#Chinchilla_scaling_(Hoffmann,_et_al,_2022)) addresses the efficiency of training large language models by balancing model size and the amount of training data. According to this law, for a given compute budget, the best performance is achieved by using smaller models trained on more data, rather than larger models with less data. This contrasts with previous approaches that focused on scaling up model size alone. The Chinchilla law suggests that for optimal performance and cost-effectiveness, a balance between model size and data quantity is essential.
```

## Where will we go from here?

Quite honestly, it's hard to predict where we'll go from here. Perhaps we will overcome the data problem by using non-textual data more effectively, or maybe we'll find ways to generate synthetic data that can help train models more efficiently (albeit, to me this kind of sounds non-sensical). The future of AI scaling is uncertain, but one thing is clear: we need to address the challenges posed by data quality, diversity, and sample efficiency to continue pushing the boundaries.
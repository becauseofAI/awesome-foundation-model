# <p align="center">Awesome Foundation Model</p>  
<div align="center"><img src="assets/sam-demo.gif"/></div>  

## Vision Large Model

<details open>
<summary>SAM: Segment Anything Model</summary>
<div align="center"><img src="assets/sam.png"/></div>    
<div align="justify">
<p>
We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation. Using our efficient model in a data collection loop, we built the largest segmentation dataset to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images. The model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions and tasks. We evaluate its capabilities on numerous tasks and find that its zero-shot performance is impressive – often competitive with or even superior to prior fully supervised results. We are releasing the Segment Anything Model (SAM) and corresponding dataset (SA-1B) of 1B masks and 11M images at https://segment-anything.com to foster research into foundation models for computer vision.
 
paper: https://arxiv.org/pdf/2304.02643.pdf  
code: https://segment-anything.com  
project: https://segment-anything.com  
demo: https://segment-anything.com/demo  
</p>
</div>
</details>

<details>
<summary>SEEM: Segment Everything Everywhere All at Once</summary>
<div align="center"><img src="assets/seem.png"/></div>    
<div align="justify">
<p>
Despite the growing demand for interactive AI systems, there have been few comprehensive studies on human-AI interaction in visual understanding e.g. segmentation. Inspired by the development of prompt-based universal interfaces for LLMs, this paper presents SEEM, a promptable, interactive model for Segmenting Everything Everywhere all at once in an image. SEEM has four desiderata: i) Versatility by introducing a versatile prompting engine for different types of prompts, including points, boxes, scribbles, masks, texts, and referred regions of another image; ii) Compositionality by learning a joint visual-semantic space for visual and textual prompts to compose queries on the fly for inference as shown in Fig. 1; iii) Interactivity by incorporating learnable memory prompts to retain dialog history information via mask-guided cross-attention; and iv) Semantic-awareness by using a text encoder to encode text queries and mask labels for open-vocabulary segmentation. A comprehensive empirical study is performed to validate the effectiveness of SEEM on various segmentation tasks. SEEM shows a strong capability of generalizing to unseen user intents as it learned to compose prompts of different types in a unified representation space. In addition, SEEM can efficiently handle multiple rounds of interactions with a lightweight prompt decoder. The SEEM demo is available at https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once, and the source code will be released at the same place.

paper: https://arxiv.org/pdf/2304.06718.pdf  
code: https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once  
demo: https://huggingface.co/spaces/xdecoder/SEEM   
</p>
</div>
</details>


##  Self-Supervised Pretraining for Vision Large Model

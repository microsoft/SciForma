import os
"""
DPO Full Fine-tune — Longshort scored pairs (0402 VAE parquet) on B200 x8
===========================================================================

K8s/Kubectl path convention:
  /data/yuxuanluo/...

Strict subset:
  score_gap >= 0.50
  winner_score >= 0.80
  loser_score <= 0.30

Expected pairs after filtering:
  4,396
"""

_base_ = ['./_dpo_base_v1.py']

# DPO objective
dpo_beta = 2000.0
dpo_sft_weight = 0.0

# Timestep sampling
weighting_scheme = "logit_normal"

# Override init checkpoints to kubectl path
policy_init_path = (
    os.environ.get("SCIFORMA_DATA_ROOT", "") + "/experiments/260216_stage2_mixed_gen_edit_b200_uniform_12wstep"
    "/checkpoint-90000/ema_weights.pt"
)
ref_init_path = policy_init_path

# Dataset
# NOTE: parquet stores absolute /mnt/data paths, so remap to /data/yuxuanluo.
dataset_cfg = dict(
    type='ArXiVParquetDatasetDPO',
    base_dir=os.environ.get('SCIFORMA_DATA_ROOT', '/data/yuxuanluo'),
    parquet_base_path='ArXiV_parquet/0402_longshort_dpo_vae',
    parquet_glob='dpo_shard_*.parquet',
    num_workers=8,
    debug_mode=False,
    is_main_process=True,
    stat_data=True,
    min_bucket_samples=0,
    path_remapping={'/mnt/data/': '/data/yuxuanluo/'},
    deterministic_latents=True,
    min_score_gap=0.50,
    min_winner_score=0.80,
    max_loser_score=0.30,
    gt_winner_threshold=0.0,
)

# Training scale (GA kept same as validated run)
train_batch_size = 1
gradient_accumulation_steps = 4
max_train_steps = 5000

learning_rate = 1e-6
lr_warmup_steps = 50
gradient_checkpointing = True

# EMA
use_ema = True

# Output / logging
model_output_dir = os.environ.get("SCIFORMA_DATA_ROOT", "") + "/experiments/260402_dpo_longshort_gap05_w08_l03_lognorm_ga4_5k_b2008"
tracker_run_name = "dpo_longshort_gap05_w08_l03_lognorm_ga4_5k_b2008"
logging_steps = 1

# Save / validation cadence
checkpointing_steps = 500
checkpoints_total_limit = 5
validation_steps = 500

# Validation setup pinned to prior OpenSciDraw-DPO setting
validation_func = 'Flux2Klein_fulltune_validation_func_parquet'

validation_prompts = [
    """The figure illustrates a human parsing pipeline composed of four main stages: Segment Generation, Segment Feature Extraction, Segment Ranking and Selection, and Part Assembling by And-or Graph. The global layout is left-to-right, with data flowing from an input image of a person on the far left through successive processing modules to produce a segmented human body map on the far right. The input image is processed in two parallel streams: one for segment generation using image segmentation techniques, and another for pose estimation, which provides skeletal keypoints used for pose-based feature extraction.

In the Segment Generation module, the input image is divided into multiple candidate segments, visually represented as small, irregularly shaped regions (e.g., parts of clothing or body limbs), shown in grayscale with blue outlines. These segments are then passed to the Segment Feature Extraction module, where three types of features are computed for each segment: traditional appearance features (represented by vertical bars with black stripes), deep-learned features (vertical bars with pink stripes), and pose-based features (vertical bars with orange stripes). The pose-based features are derived from a separate stream that first estimates the human pose (shown as a stick figure overlaid on the input image with colored joints and limbs) and then extracts features based on spatial relationships between the segments and the pose keypoints.

The extracted features are then used in the Segment Ranking and Selection module. Here, segments are ranked and selected based on their feature quality; top-ranked segments are enclosed in red dashed boxes, while lower-ranked ones are in gray dashed boxes. A purple downward arrow indicates the selection process, filtering out less relevant segments.

The selected segments are then fed into the Part Assembling by And-or Graph module. This module uses a hierarchical graph structure to assemble the final human parsing result. At the top, a 'human' node branches into 'human configuration 1' and other configurations. Each configuration decomposes into upper-body and lower-body components, which further break down into specific parts like head & torso, left arm, right arm, etc. For each part, multiple candidate segments (represented by geometric shapes like stars, triangles, and circles in various colors) are considered. The graph structure allows for flexible assembly via AND (all required parts) and OR (any one of several candidates) logic. The final assembled result is a color-coded human body segmentation map, where different body parts are labeled with distinct colors (e.g., blue for hair, green for upper clothes, yellow for lower clothes, etc.), as indicated in the legend at the bottom right. Dashed lines connect the selected segments back to the graph, showing how they are assigned to specific body parts. The entire pipeline is designed to progressively refine and assemble a coherent human parsing output from raw image data.""",
    """The figure presents two side-by-side diagrams, labeled (a) and (b), illustrating the application of a proposed method in convolutional neural networks (CNNs) using stacked 'quasi-hexagonal' kernels. Both diagrams depict a three-layered CNN structure, with layers labeled as Layer l, Layer l+1, and Layer l+2, arranged vertically from bottom to top. Each layer is represented as a grid of square units, symbolizing feature maps or activation maps, with multiple such grids stacked horizontally to indicate depth or channels.

In diagram (a), the kernel pattern used in Layer l is labeled 'L' and is visually represented by a green rectangular region within the grid, surrounded by an orange border indicating the receptive field. In Layer l+1, the kernel pattern is labeled 'U', shown as a blue rectangular region. The output neuron N(i,j,c) in Layer l+2 is highlighted in red, and arrows connect it to the corresponding receptive fields across the layers, demonstrating the convolution operation followed by non-linearity. The receptive fields for the neuron are explicitly labeled and span across all three layers, showing how the kernel patterns stack to form the final receptive field.

Diagram (b) mirrors the structure of (a) but swaps the kernel patterns between layers: Layer l uses kernel pattern 'U' (blue region), while Layer l+1 uses kernel pattern 'L' (green region). Despite this reversal, the resulting receptive field for the neuron N(i,j,c) in Layer l+2 remains identical in shape and extent to that in (a), as indicated by the consistent red highlight and connecting arrows. This demonstrates that the final receptive field pattern is invariant to the order of kernel application, provided the same set of kernel patterns is used across the layers.

Both diagrams include labels for 'convolution + non-linearity' pointing to the connections between layers, emphasizing the standard CNN processing steps. The kernel patterns 'L' and 'U' are visually distinct in color (green and blue respectively) and are positioned centrally within each layer's grid. The receptive fields are outlined with black lines and shaded in orange, clearly demarcating the spatial extent of influence for the neuron. The overall layout is symmetrical and hierarchical, with clear visual flow from input (Layer l) through intermediate (Layer l+1) to output (Layer l+2). The figure caption reinforces that the method employs small-size quasi-hexagonal kernels and highlights the key insight: the induced receptive field pattern remains unchanged regardless of the order in which the same kernel patterns are applied across layers.""",
    """The figure presents four distinct methodologies (labeled a, b, c, d) for training deep convolutional neural networks (CNNs) to classify 2D radiology images into different topic categories, using text from radiology reports as supervisory signals. Each panel illustrates a different level of text granularity and topic modeling approach.

In panel (a), the process begins with a set of documents (radiology reports), which undergo Natural Language Processing (NLP) to extract images mentioned within them (~216k images). These documents are then subjected to LDA (Latent Dirichlet Allocation) document topic clustering, resulting in a set of high-level topics (e.g., topic 0 with keywords 'protocol, abdomen', topic 2 with 'kidney, renal', etc.). Each topic is associated with its corresponding images. These image-topic pairs are used to train a deep CNN to classify 2D radiology images into these document-level topic categories. The CNN is depicted as a vertical, light green gradient rectangle labeled 'deep convolutional neural network'.

Panel (b) extends this approach by introducing hierarchical topic modeling. After initial LDA document topic clustering, each document topic is further subdivided into 10 sub-topics using LDA topic modeling on each document topic cluster (h2-LDA). For example, topic 0 splits into sub-topics 0-0 through 0-9, and similarly for other topics. These sub-topics are then associated with images, and the resulting image-subtopic pairs are used to train a deep CNN (shown as a vertical, light orange gradient rectangle) to classify images into finer-grained sub-topic categories.

Panel (c) focuses on sentence-level topic modeling. Documents are processed via NLP to extract sentences mentioning findings and their preceding/following sentences. These sentences are then clustered using LDA at the sentence level, producing a large number of sentence-level topics (e.g., topic 0 with 'radiographically', topic 1 with 'perinephric', up to topic 999 with 'seroma'). Images mentioned in these sentences (~187k images) are associated with their respective sentence-level topics. This association is used to train a deep CNN (vertical, light purple gradient rectangle) to classify images based on sentence-level topics.

Panel (d) introduces an image-to-word model. Radiology reports are processed via NLP to extract sentences mentioning findings. These sentences are linked to images mentioned within them. A disease ontology is used to identify two disease-related terms per sentence (e.g., 'which node'). A recurrent neural network maps words to vectors, creating a word2vec representation. The text-vector output layer of a pre-trained image-to-sentence-level-topic CNN is fine-tuned to output a vector closest to the target vector representing the disease in the image. This fine-tuned model is then used to perform topic clustering of images based on sentence-level LDA, and the resulting image-topic pairs are used to train another deep CNN (vertical, light purple gradient rectangle) to classify images into topic categories. All panels show a consistent flow: text processing → topic modeling → image association → CNN training, with arrows indicating data flow and training direction.""",
    """The figure presents a complete flow diagram of the Recursive Context Propagation Network (RCPN) for semantic segmentation, structured into two main stages: Visual Feature Extraction and Recursive Context Propagation Network (RCPN), separated by a dashed vertical boundary.

[1] Global Layout and Structure:
The diagram is horizontally organized from left to right, depicting the data flow through the network. The left section, labeled 'Visual Feature Extraction', processes an input image I to generate visual features V. The right section, labeled 'Recursive Context Propagation Network (RCPN)', performs iterative context propagation and refinement using these features. The final output is compared against ground truth for error computation.

[2] Visual Modules and Attributes:
In the Visual Feature Extraction stage, an input image I (a sailboat scene) passes through an orange rectangular module labeled F_CNN, representing a convolutional neural network. Its output is combined with a gray 3D block labeled V (visual feature map) via a summation operator (+). Below this, a colorful segmented image labeled 'superpixels' indicates region-based processing. From this stage, multiple visual feature streams v_1, v_2, ..., v_s are extracted and fed into the RCPN.

In the RCPN stage, each v_i enters a light blue rectangular module labeled F_sem (semantic feature extractor). Outputs x_1, x_2, ..., x_s from these modules feed into a recursive structure. Each x_i connects to a pink rectangular module labeled F_com (context combiner), which merges features across levels. For example, x_1 feeds into F_com to produce x_12; x_2 also feeds into F_com to produce x_3; and x_12 and x_3 feed into another F_com to produce x_123, illustrating recursive combination. These combined features then pass through light green rectangular modules labeled F_dec (decoder), generating reconstructed outputs x̃_1, x̃_12, x̃_3, ..., x̃_s. Each reconstructed output is processed by a purple rectangular module labeled F_lab (label classifier) to produce a predicted segmentation.

A red dashed arrow labeled 'Bypass error' runs from x̃_1 back to x_1, indicating a feedback mechanism for error propagation. On the far right, the predicted segmentations are compared with ground truth images (two versions shown: one with green hill, one with green sail) via a red circular subtraction operator (−), producing a 'Classifier Error' signal, highlighted in red text and arrows.

[3] Connections and Arrows:
Solid black arrows indicate forward propagation: from I → F_CNN → V → + → v_i → F_sem → x_i → F_com → x_ij → F_dec → x̃_ij → F_lab → Predicted. The recursive structure uses branching connections: x_1 and x_2 both feed into F_com to form x_12; x_12 and x_3 feed into another F_com to form x_123, etc. A red dashed arrow labeled 'Bypass error' loops from x̃_1 back to x_1, suggesting error feedback. Red solid arrows connect the predicted outputs and ground truth to the subtraction node, which outputs the 'Classifier Error'. This error signal is visually emphasized with bold red lines and text, indicating its role in training or refinement.""",
    """The figure presents an overview of the Horde architecture, a framework for learning an ensemble of reward shapings in reinforcement learning. The global layout is structured as a flowchart with distinct modules arranged from top-left to bottom-right, depicting a data and control flow from the environment through shaping, scaling, function approximation, and finally to an ensemble policy. The top-left corner contains a green oval labeled 'Behavior' enclosing an 'Environment' ellipse, indicating the interaction loop between state s, action a, and next state s'. This environment outputs the raw reward R and the transition tuple (s, a, s') to downstream components.

The main processing pipeline begins with the 'Shaping' module, which receives the raw reward R and the current state s. A designer input, denoted by the symbol Φ, is fed into Shaping, which then produces multiple shaping functions Φ₁, ..., Φᵢ, ..., Φₗ. These are passed to the 'Scaling' module, which also receives a control signal C. The Scaling module applies scalar coefficients c¹₁, ..., cᵢⱼ, ..., cₗₖₗ to each shaping function, producing scaled rewards R, R¹₁, ..., Rᵢⱼ, ..., Rₗₖₗ. These scaled rewards are represented as bold lines, indicating they are vectors.

Below the Shaping and Scaling modules lies the 'Function Approximation' block, which takes the transition (s, a, s') as input. It outputs a set of blue horizontal lines representing feature vectors derived from the transition. These features are connected via vertical green lines labeled θ₀, θ¹₁, ..., θᵢⱼ, ..., θₗₖₗ, which represent weight parameters for individual learners. Each intersection of a green line (weight) with a blue line (feature) corresponds to a weighted feature for a specific learner.

Each learner is represented by an inverted blue triangle labeled d₀, d¹₁, ..., dᵢⱼ, ..., dₗₖₗ. Each triangle receives its corresponding scaled reward (R, R¹₁, etc.) and computes a policy π₀, π¹₁, ..., πᵢⱼ, ..., πₗₖₗ, shown as red downward arrows. These policies are then aggregated into an 'Ensemble' module, depicted as a rounded rectangle, which combines them to produce the final 'Ensemble policy'.

A feedback loop is shown: the output of the Ensemble, labeled a', is a vector of greedy actions at state s' with respect to each policy πᵢⱼ. This vector feeds back into the Environment, closing the learning loop. The caption clarifies that in this latent setting, all environmental interactions occur only in the upper-left corner, meaning the rest of the architecture operates on learned representations. The diagram uses color coding—green for weights, blue for features, red for policies, and bold black lines for reward vectors—to distinguish different types of signals.""",
    """The figure illustrates a deep learning framework for fine-grained classification using triplet-based training with shared parameters across three convolutional neural networks (CNNs). The global layout is left-to-right, starting with the input data on the far left, progressing through three parallel CNN branches, followed by normalization, structured label embedding, and finally loss computation on the right. The entire pipeline is designed to learn discriminative features and a fine-grained classifier by jointly optimizing two loss functions.

On the left, the 'Training Data' is represented as a vertical stack of horizontal bars, each colored blue, green, or red, corresponding to reference (ref), positive (pos), and negative (neg) images respectively, as indicated by the legend above. These triplets are fed into three parallel CNNs labeled R (blue), P (green), and N (red), which share the same network architecture and parameters, as denoted by the label 'Parameters Sharing CNN'. Each CNN consists of multiple fully connected layers depicted as interconnected nodes, with the final layer being a dense, fully connected layer.

The outputs from each of the three CNNs are passed through individual L2 normalization blocks, shown as gray rectangular boxes labeled 'ℓ₂', to normalize the feature vectors. These normalized features are then combined and fed into a module labeled 'Structured Label', enclosed in a purple rounded rectangle. This module visualizes the hierarchical or attribute-based structure of the labels: at the top, a row of three circles (blue, green, red) represents the triplet classes; below, two groups of three circles each (with overlapping colors) indicate shared or hierarchical relationships; further down, four geometric shapes (hexagon, star, square, star) are connected to the lower label groups, symbolizing higher-level semantic or structural attributes.

From the 'Structured Label' module, two loss components are computed. The first is 'Softmax with Loss', shown in an orange box, which computes the standard classification loss. The second is 'Generalized Triplet Loss', also in an orange box, which enforces discriminative feature learning based on the triplet structure. A plus sign between these two boxes indicates that they are summed together for joint optimization. The final block, labeled 'Loss Computation', signifies the overall objective function that combines both losses to train the model. The entire process is designed to simultaneously learn robust feature representations and a fine-grained classifier by leveraging both class labels and their underlying structural relationships.""",
]

resolution_list = [
    [1472, 704],
    [1600, 640],
    [1088, 960],
    [1728, 576],
    [1024, 1024],
    [1920, 512],
]

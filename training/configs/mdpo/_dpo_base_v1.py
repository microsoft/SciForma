import os
"""
DPO Training Configuration for Flux2Klein 9B
=============================================

Trains the stage-2 policy model using DPO on 50k winner/loser pairs.

Hardware target:
  4x A100 80GB GPUs, DeepSpeed ZeRO-2

Key DPO hyperparameters:
  dpo_beta      : temperature of the DPO objective (2000 typical for diffusion)
  dpo_sft_weight: weight of auxiliary SFT regularisation term on winners (0 = off)
  ref_on_cpu    : keep frozen reference on CPU to save ~18GB VRAM (slower)

Usage:
    accelerate launch --config_file accelerate_cfg/deepspeed_zero2_bf16.yaml \\
        train_OpenSciDraw_dpo.py configs/260320_dpo/flux2klein_dpo_config.py
"""

_base_ = [
    '../base_config.py',
]

# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
model_type = 'Flux2Klein'

transformer_cfg = dict(type='Flux2Transformer2DModel')

pretrained_model_name_or_path = "black-forest-labs/FLUX.2-klein-base-9B"
huggingface_token = None   # resolved from HF_TOKEN env var or _local_secrets.py

use_lora = False           # Full fine-tuning (no LoRA)
lora_layers = None

# ──────────────────────────────────────────────────────────────────────────────
# DPO checkpoint paths
# ──────────────────────────────────────────────────────────────────────────────
# Path to the stage-2 EMA weights (used for BOTH policy init and reference).
# After DPO starts, the policy diverges from reference while reference stays frozen.
policy_init_path = (
    os.environ.get("SCIFORMA_DATA_ROOT", "") + "/experiments/260216_stage2_mixed_gen_edit_b200_uniform_12wstep"
    "/checkpoint-90000/ema_weights.pt"
)
ref_init_path = policy_init_path      # same as policy init = start from same weights

# Memory trade-off:
#   ref_on_cpu=False  → reference on GPU (faster but +18 GB VRAM per GPU)
#   ref_on_cpu=True   → reference on CPU (slower due to CPU↔GPU transfers, saves 18 GB)
ref_on_cpu = True

# ──────────────────────────────────────────────────────────────────────────────
# DPO Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
# Beta controls the strength of the DPO constraint:
#   higher beta → policy stays closer to reference (more conservative)
#   lower  beta → policy changes more aggressively
# For diffusion models the MSE magnitude is small, so beta is typically large.
# Typical range for Flux-style models: 500–5000
dpo_beta = 2000.0

# SFT regularisation on winner MSE.  0 = pure DPO.  Try 0.01 if policy collapses.
dpo_sft_weight = 0.0

# ──────────────────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average)
# ──────────────────────────────────────────────────────────────────────────────
use_ema = True
ema_decay = 0.9999
ema_steps = 100           # update EMA every N optimizer steps
ema_on_gpu = False        # CPU-based EMA to save VRAM (A100 80G)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
use_parquet_dataset = True

# Adjust paths once you have run build_dpo_parquet.py
dataset_cfg = dict(
    type='ArXiVParquetDatasetDPO',
    base_dir=os.environ.get('SCIFORMA_DATA_ROOT', '') + '',
    parquet_base_path='experiments/260320_dpo_parquet',  # output of build_dpo_parquet.py
    # Only load final merged shards; ignore partial_shard_* checkpoints (which are duplicates)
    parquet_glob='dpo_shard_*.parquet',
    num_workers=4,
    debug_mode=False,
    is_main_process=True,
    stat_data=False,
    min_bucket_samples=0,
    # On AMLT, parquet paths are already /mnt/data/... — no remapping needed.
    # For local testing, override with: {os.environ.get('SCIFORMA_DATA_ROOT', '') + '/': '<your_data_path>/'}
    path_remapping={},
    # DPO requires deterministic x0 (VAE mean, not a reparameterized sample).
    # Stochastic sampling introduces per-step noise in gradient signal.
    deterministic_latents=True,
)

sampler_cfg = dict(
    type='DistributedBucketSamplerV2',
    dataset=None,          # filled in by training script
    batch_size=1,
    num_replicas=1,        # accelerator handles GPU distribution
    rank=0,
    drop_last=True,
    shuffle=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Training iteration function
# ──────────────────────────────────────────────────────────────────────────────
train_iteration_func = 'Flux2Klein_dpo_train_iteration'

# ──────────────────────────────────────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
train_batch_size = 1                    # per GPU; effective = batch*gpus*grad_accum
gradient_accumulation_steps = 8        # effective batch = 1*4*8=32 pairs
max_train_steps = 5000
num_train_epochs = 100                  # epoch count is a ceiling; steps win

# ──────────────────────────────────────────────────────────────────────────────
# Optimizer
# ──────────────────────────────────────────────────────────────────────────────
optimizer = "AdamW"
use_8bit_adam = False          # incompatible with DeepSpeed

# DPO typically needs a smaller LR than full fine-tuning
learning_rate = 5e-7
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 1e-2
adam_epsilon = 1e-08

# ──────────────────────────────────────────────────────────────────────────────
# LR Schedule
# ──────────────────────────────────────────────────────────────────────────────
lr_scheduler = "cosine"
lr_warmup_steps = 200

# ──────────────────────────────────────────────────────────────────────────────
# Gradient & Mixed Precision
# ──────────────────────────────────────────────────────────────────────────────
max_grad_norm = 1.0
mixed_precision = "bf16"
allow_tf32 = True
gradient_checkpointing = True

# ──────────────────────────────────────────────────────────────────────────────
# Flow-matching noise weighting (inherited from stage-2)
# ──────────────────────────────────────────────────────────────────────────────
weighting_scheme = "none"
logit_mean = 0.0
logit_std  = 1.0
mode_scale = 1.29
validation_guidance_scale = 4.0

# ──────────────────────────────────────────────────────────────────────────────
# Guidance scale for DPO forward passes
# ──────────────────────────────────────────────────────────────────────────────
guidance_scale = 4.0

# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing & Logging
# ──────────────────────────────────────────────────────────────────────────────
checkpointing_steps  = 200
checkpoints_total_limit = 5            # keep the 5 most recent checkpoints
resume_from_checkpoint = "latest"

model_output_dir = os.environ.get("SCIFORMA_DATA_ROOT", "") + "/experiments/260325_dpo_fulltune_beta500_ema_v2"
logging_dir = "logs"

# ──────────────────────────────────────────────────────────────────────────────
# WandB
# ──────────────────────────────────────────────────────────────────────────────
report_to = "wandb"
wandb_project = "OpenSciDraw-DPO"
tracker_run_name = "dpo_fulltune_beta500_lr5e7_ema"
logging_steps = 10
verbose_logging = True

# ──────────────────────────────────────────────────────────────────────────────
# DataLoader
# ──────────────────────────────────────────────────────────────────────────────
dataloader_num_workers = 4
seed = 42

# ──────────────────────────────────────────────────────────────────────────────
# Validation (generate sample images to monitor quality)
# ──────────────────────────────────────────────────────────────────────────────
max_sequence_length = 2048  # match GT NPZ encoding (2048 max tokens)

validation_steps = 200          # run validation at every checkpoint
validation_func = 'Flux2Klein_fulltune_validation_func_parquet'

validation_prompts = [
    """The figure illustrates a human parsing pipeline composed of four main stages: Segment Generation, Segment Feature Extraction, Segment Ranking and Selection, and Part Assembling by And-or Graph. The global layout is left-to-right, with data flowing from an input image of a person on the far left through successive processing modules to produce a segmented human body map on the far right. The input image is processed in two parallel streams: one for segment generation using image segmentation techniques, and another for pose estimation, which provides skeletal keypoints used for pose-based feature extraction.

In the Segment Generation module, the input image is divided into multiple candidate segments, visually represented as small, irregularly shaped regions (e.g., parts of clothing or body limbs), shown in grayscale with blue outlines. These segments are then passed to the Segment Feature Extraction module, where three types of features are computed for each segment: traditional appearance features (represented by vertical bars with black stripes), deep-learned features (vertical bars with pink stripes), and pose-based features (vertical bars with orange stripes). The pose-based features are derived from a separate stream that first estimates the human pose (shown as a stick figure overlaid on the input image with colored joints and limbs) and then extracts features based on spatial relationships between the segments and the pose keypoints.

The extracted features are then used in the Segment Ranking and Selection module. Here, segments are ranked and selected based on their feature quality; top-ranked segments are enclosed in red dashed boxes, while lower-ranked ones are in gray dashed boxes. A purple downward arrow indicates the selection process, filtering out less relevant segments.

The selected segments are then fed into the Part Assembling by And-or Graph module. This module uses a hierarchical graph structure to assemble the final human parsing result. At the top, a 'human' node branches into 'human configuration 1' and other configurations. Each configuration decomposes into upper-body and lower-body components, which further break down into specific parts like head & torso, left arm, right arm, etc. For each part, multiple candidate segments (represented by geometric shapes like stars, triangles, and circles in various colors) are considered. The graph structure allows for flexible assembly via AND (all required parts) and OR (any one of several candidates) logic. The final assembled result is a color-coded human body segmentation map, where different body parts are labeled with distinct colors (e.g., blue for hair, green for upper clothes, yellow for lower clothes, etc.), as indicated in the legend at the bottom right. Dashed lines connect the selected segments back to the graph, showing how they are assigned to specific body parts. The entire pipeline is designed to progressively refine and assemble a coherent human parsing output from raw image data.""",
    """The figure presents two side-by-side diagrams, labeled (a) and (b), illustrating the application of a proposed method in convolutional neural networks (CNNs) using stacked 'quasi-hexagonal' kernels. Both diagrams depict a three-layered CNN structure, with layers labeled as Layer l, Layer l+1, and Layer l+2, arranged vertically from bottom to top. Each layer is represented as a grid of square units, symbolizing feature maps or activation maps, with multiple such grids stacked horizontally to indicate depth or channels.

In diagram (a), the kernel pattern used in Layer l is labeled 'L' and is visually represented by a green rectangular region within the grid, surrounded by an orange border indicating the receptive field. In Layer l+1, the kernel pattern is labeled 'U', shown as a blue rectangular region. The output neuron N(i,j,c) in Layer l+2 is highlighted in red, and arrows connect it to the corresponding receptive fields across the layers, demonstrating the convolution operation followed by non-linearity. The receptive fields for the neuron are explicitly labeled and span across all three layers, showing how the kernel patterns stack to form the final receptive field.

Diagram (b) mirrors the structure of (a) but swaps the kernel patterns between layers: Layer l uses kernel pattern 'U' (blue region), while Layer l+1 uses kernel pattern 'L' (green region). Despite this reversal, the resulting receptive field for the neuron N(i,j,c) in Layer l+2 remains identical in shape and extent to that in (a), as indicated by the consistent red highlight and connecting arrows. This demonstrates that the final receptive field pattern is invariant to the order of kernel application, provided the same set of kernel patterns is used across the layers.

Both diagrams include labels for 'convolution + non-linearity' pointing to the connections between layers, emphasizing the standard CNN processing steps. The kernel patterns 'L' and 'U' are visually distinct in color (green and blue respectively) and are positioned centrally within each layer’s grid. The receptive fields are outlined with black lines and shaded in orange, clearly demarcating the spatial extent of influence for the neuron. The overall layout is symmetrical and hierarchical, with clear visual flow from input (Layer l) through intermediate (Layer l+1) to output (Layer l+2). The figure caption reinforces that the method employs small-size quasi-hexagonal kernels and highlights the key insight: the induced receptive field pattern remains unchanged regardless of the order in which the same kernel patterns are applied across layers.""",
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
    """The figure presents two side-by-side diagrams labeled (a) 'No pruning' and (b) 'Pruning by bound', illustrating a tree-based search or optimization process with a focus on pruning strategies. Both diagrams depict a binary tree structure rooted at node η₀, which has an associated loss value ℒ = 0. Each node is represented as a circle containing the node identifier (η₀, η₁, ..., η₆), with its corresponding loss value ℒ displayed to the right of the node. The tree branches via directed edges labeled e₁ and e₂, indicating different choices or paths from each parent node.

In diagram (a), all nodes are explored without pruning. The path from η₀ → η₁ → η₃ is highlighted with dashed red lines along the edges, indicating it is the current exploration path. Node η₃ has ℒ = 7, and a dotted red circle surrounds it, possibly emphasizing its role as a leaf or terminal state being evaluated. Other leaf nodes have ℒ values of 9 (η₄), 11 (η₅), and 10 (η₆).

Diagram (b) demonstrates pruning by bound. The same tree structure is shown, but now the edge from η₀ to η₂ is crossed out with a thick blue diagonal line, indicating that this branch is pruned. This pruning occurs because the loss value at η₂ increases to ℒ = 9, which exceeds the bound established by the explored path η₀ → η₁ → η₃ (ℒ = 7). As a result, further exploration of the subtree rooted at η₂ is avoided. The path η₀ → η₁ → η₃ remains highlighted with dashed red lines, and η₃ still has ℒ = 7. The other leaf nodes retain their original ℒ values: 9 (η₄), 11 (η₅), and 10 (η₆).

Between the two diagrams, vertical dotted arrows labeled p₁ and p₂ indicate the sequence of processing steps or phases. In (a), p₁ points from the root to the first level, and p₂ points from the first to the second level. In (b), p₂ points from the root to the first level, and p₁ points from the first to the second level, suggesting a reordered or optimized traversal where pruning allows skipping unnecessary evaluations.

The overall layout is horizontal, with (a) on the left and (b) on the right, separated by a small gap. The caption below the figure states: 'Efficient pruning by bound can be achieved by sorting the positive windows by decreasing difficulty,' implying that the pruning strategy prioritizes exploring harder (higher-loss) paths first to establish tighter bounds early, thereby enabling more aggressive pruning of less promising branches.""",
    """The figure presents a schematic representation of constructing network models from linguistic data across languages, specifically focusing on polysemy and semantic relationships. It is divided into three parts: (a), (b), and (c).

Part (a) illustrates a tripartite network structure with three horizontal layers labeled 'Sample level: S', 'Words level: w^L', and 'Meaning level: m'. The top layer contains sample-level nodes: 'MOON' and 'SUN', represented as black-outlined ovals. These connect downward via directed arrows to word-level nodes in the middle layer, which are colored red for Coast Tsimshian and blue for Lakhota, as indicated in the legend. Red nodes include 'gyemgmáatk', 'gooypah', 'gyemk', and 'gimgmdziws'; blue nodes include 'hanjwí', 'hanhépi_wí', 'wí', and 'ánpawí'. These word-level nodes further connect downward to meaning-level nodes in the bottom layer: 'MOON', 'month', 'heat', and 'SUN', also shown as black-outlined ovals. The connections from sample to word level are labeled 'translation t_sw', and those from word to meaning level are labeled 'backtranslation t_wm', both indicated by curved orange arrows on the left side. Red and blue arrows represent the respective language pathways.

Part (b) shows a directed bipartite graph derived from part (a) by projecting the tripartite network onto the sample and meaning levels, aggregating the word-level connections. This graph includes the same sample nodes ('MOON', 'SUN') at the top and meaning nodes ('MOON', 'month', 'heat', 'SUN') at the bottom. Directed edges between them are thick black arrows with numerical weights (e.g., 6, 2, 1, 4) indicating aggregated link counts from the original tripartite network.

Part (c) displays a directed and weighted unipartite graph, obtained by further projecting the bipartite graph in (b) by merging identical sample-level nodes (i.e., 'MOON' and 'SUN' are treated as single entities). The resulting graph contains only meaning-level nodes: 'MOON', 'month', 'heat', 'SUN', connected by bidirectional or unidirectional arrows with weights (e.g., 2, 1). The edges reflect aggregated semantic relationships derived from the original translation and back-translation paths.

The overall layout is hierarchical and modular, progressing from a detailed tripartite representation to simplified bipartite and unipartite projections, emphasizing the abstraction process in network construction. All nodes are oval-shaped, and edge colors (red/blue) and weights are used to encode linguistic and structural information.""",
    """The figure illustrates the initial synaptic weight configuration for a neural network controlling a robot's movement behavior, specifically designed to enforce an attraction response. The global layout is a two-layer feedforward network with four input neurons at the top and two output motor neurons at the bottom. The top layer consists of two pairs of sensor neurons: 'Left Food Sensor Neurons' and 'Right Food Sensor Neurons' on the left side, and 'Left Container Neurons' and 'Right Container Neurons' on the right side. These are represented as light green rectangular boxes with black borders and black text. The bottom layer contains two motor neurons: 'Left Motor Neurons' and 'Right Motor Neurons', also depicted as light green rectangles with black borders and black text. All connections between the top and bottom layers are shown as arrows indicating directed influence. Solid black arrows represent synapses set to maximum strength, while dashed blue arrows indicate synapses set to zero. Specifically, each food sensor neuron connects with a solid arrow to the motor neuron on the same side (e.g., Left Food Sensor Neurons → Left Motor Neurons), and each container neuron connects with a solid arrow to the opposite-side motor neuron (e.g., Left Container Neurons → Right Motor Neurons). Additionally, cross-connections from food sensors to the opposite motor neuron and from container neurons to the same-side motor neuron are shown as dashed blue lines, signifying they are disabled (set to zero). A legend box on the far right, outlined in black, clarifies the meaning of the arrow types: a solid black downward arrow labeled 'Synapses Set to Maximum' and a dashed blue downward arrow labeled 'Synapses Set to Zero'. This setup ensures that food detection activates the corresponding side’s motor, promoting movement toward food, while container detection activates the opposite motor, causing the robot to turn away from containers. The overall structure is symmetrical and clearly organized to reflect the intended behavioral logic: attraction to food and avoidance of containers.""",
    """The figure presents an end-to-end neural network architecture for image captioning, structured into two main horizontal pathways: a top-level processing flow and a detailed Localization Layer at the bottom. The global layout begins on the left with an input image of dimensions 3xWxH, depicted as a photograph of two cats watching a TV screen. This image feeds into a gray trapezoidal block labeled 'CNN', which outputs convolutional features represented as a light blue cube with dimensions CxW'xH'.

From the CNN output, the top pathway proceeds through a vertical purple rectangular block, symbolizing further feature processing, leading to region features shown as stacked green and blue cubes with dimensions BxCxXxY. These are then passed through a 'Recognition Network' (represented by three vertical black bars), which produces region codes of size BxD. These region codes are fed into an LSTM (gray rectangle), which generates descriptive captions such as 'Striped gray cat' (green box) and 'Cats watching TV' (blue box), displayed alongside a cropped image with bounding boxes.

The bottom half of the diagram details the 'Localization Layer', enclosed in a large lavender rounded rectangle. It takes the same CNN output (light blue cube, CxW'xH') as input. Inside this layer, a gray trapezoid labeled 'Conv' processes the features to produce 'Region Proposals' (4kxW'xH'), visualized as a grid with a star indicating a selected region. Concurrently, 'Region scores' (kxW'xH') are generated from the Conv output. These scores guide the selection of 'Best Proposals' (Bx4), shown as two overlapping rectangles (one green, one blue), via a 'Sampling' step.

The Best Proposals are then fed into a 'Grid Generator', which creates a 'Sampling Grid' (BxXxYx2), illustrated as a grid of dots. This grid, along with the original conv features, is input into a 'Bilinear Sampler' (a gray circle with an 'X' inside), which extracts the final 'Region features' (Bx512x7x7), shown as stacked green and blue cubes. These features match those in the top pathway and feed into the Recognition Network.

Connections are indicated by solid black arrows showing data flow: from image to CNN, through the Localization Layer components, and up to the LSTM. A dashed line connects the top pathway's Recognition Network to the Localization Layer, emphasizing its role in generating region features. The LSTM has a feedback loop arrow pointing back to itself, indicating recurrent processing. All blocks are labeled with their function or data dimension, and the entire model is trained end-to-end with gradient descent.""""",
    """The figure presents an overview of a joint image-text topic detection and tracking framework for news content. The global layout is structured as a top-down pipeline divided into three main stages: preprocessing, topic detection, and topic tracking, separated by dashed horizontal lines. At the top, two input sources are shown: 'News Videos', represented by a collage of video frames depicting news anchors and scenes, and 'News Captions', shown as a block of sample text from news broadcasts. Both inputs flow downward via blue arrows into a rectangular box labeled 'Preprocessing: Segmentation', indicating initial processing steps such as splitting videos and texts into manageable segments.

Following preprocessing, the pipeline enters the 'Topic Detection' stage, enclosed in a large rounded rectangle. This stage is titled 'Joint Image-Text Topic Detection with And-Or Graph Representation by Swendsen-Wang Cuts Cluster Sampling'. It illustrates the detection of topics across multiple time periods, labeled 'Topics Detected in Time Period 1' and 'Topics Detected in Time Period M', with ellipses suggesting intermediate periods. Each time period contains one or more topics (e.g., 'Topic 1-1', 'Topic M-1'), each represented as an And-Or graph. These graphs have circular nodes: the root node splits into 'Text' and 'Image' branches. The 'Text' branch further divides into 'Who', 'Where', and 'What' sub-nodes, while the 'Image' branch splits into 'Face' and 'Object'. Each of these leaf nodes connects to small square boxes representing specific entities or features. Red dashed lines connect related entities across the text and image branches, indicating joint semantic associations. The entire structure under this stage is labeled 'Detected Topics'.

The output of topic detection feeds into the next stage, 'Topic Tracking', also enclosed in a rounded rectangle. This stage is labeled 'Joint Image-Text Topic Tracking'. It visualizes the temporal evolution of topics using a sequence of circular nodes arranged horizontally. Multiple colored paths (blue, orange, green, purple) connect these nodes, illustrating how individual topics evolve, merge, split, or disappear over time. These paths represent 'Topic Trajectories', as labeled at the bottom right. The figure uses consistent visual attributes: blue arrows denote data flow; circular nodes represent topics or topic components; square nodes represent atomic features; and dashed red lines indicate cross-modal associations. The overall design emphasizes the integration of visual and textual modalities for coherent topic modeling and longitudinal tracking.""",
    '''The figure illustrates the five-step BLFRB (Bootstrap-based Localized Fixed-Point Resampling Bootstrap) procedure, structured as a horizontal workflow divided into five sequential stages, labeled step 1 through step 5, separated by vertical dashed lines. The entire process is organized into two parallel processing modules, one for each subsample, both sharing the same structure and operating independently. These modules are visually grouped within light yellow rectangular backgrounds.

[1] Global Layout and Structure:

The diagram begins on the far left with the original dataset X = (x₁ … xₙ), represented as a rounded rectangle. From this, two separate paths diverge, each leading to a distinct subsample module. Each module follows the same five-step sequence: Step 1 involves generating a subsample; Step 2 generates multiple bootstrap samples from that subsample; Step 3 computes resampled parameter estimates using the initial estimate from Step 1; Step 4 aggregates these resampled estimates into a subsample-level uncertainty estimate; and Step 5 combines the results from all subsample modules into a final overall uncertainty estimate. The two modules are vertically aligned, with the top module corresponding to the first subsample (denoted with superscript (1)) and the bottom module to the s-th subsample (denoted with superscript (s)). A red feedback loop connects Step 3 to Step 1 within each module, indicating an iterative or recursive update of the initial estimate.

[2] Visual Modules and Attributes:

Each module contains nodes represented as rounded rectangles with black borders. In Step 1, the node Ĥ⁽¹⁾ = (x̂₁⁽¹⁾ … x̂ᵇ⁽¹⁾) represents the first subsample of size b, and similarly for Ĥ⁽ˢ⁾ in the lower module. From each subsample, a downward arrow leads to the initial estimate ˆθₙ,ᵦ⁽¹⁾ (or ˆθₙ,ᵦ⁽ˢ⁾), which is used in subsequent steps. In Step 2, multiple bootstrap samples are generated, denoted as X*(11), X*(12), ..., X*(1r) in the top module, and X*(s1), X*(s2), ..., X*(sr) in the bottom module. Each bootstrap sample is defined as (Ĥ⁽¹⁾; n*(1j)) or (Ĥ⁽ˢ⁾; n*(sj)), where n* denotes the bootstrap indices. In Step 3, each bootstrap sample feeds into a resampled parameter estimate ˆθₙ,ᵦᴿ*(1j) or ˆθₙ,ᵦᴿ*(sj), shown as a node with a superscript R* indicating resampling. In Step 4, these resampled estimates converge to a subsample-level uncertainty estimate, denoted as ξ̂*(1) and ξ̂*(s). Finally, in Step 5, the final overall uncertainty estimate ξ̂* is computed as the average over all subsample modules: ξ̂* = (1/s) Σᵢ=₁ˢ ξ̂*(i), displayed as a large node on the far right.

[3] Connections and Arrows:

All connections are solid black arrows, except for the red arrows indicating feedback loops. From the original data X, two black arrows point to the first subsample Ĥ⁽¹⁾ and the s-th subsample Ĥ⁽ˢ⁾. Within each module, black arrows flow from the subsample to the initial estimate, then from the initial estimate to each bootstrap sample, and from each bootstrap sample to its corresponding resampled estimate. From each resampled estimate, a black arrow points to the subsample’s uncertainty estimate ξ̂*(1) or ξ̂*(s). Red arrows form a vertical feedback loop from each resampled estimate ˆθₙ,ᵦᴿ* back to the initial estimate ˆθₙ,ᵦ, suggesting that the resampled estimates may be used to refine the initial estimate in an iterative fashion. Finally, black arrows from each ξ̂*(i) point to the final average ξ̂*, completing the workflow. The red feedback loops are emphasized to highlight the recursive nature of the algorithm within each subsample module.''',
]

resolution_list = [
    [1472, 704],
    [1600, 640],
    [1088, 960],
    [1728, 576],
    [1024, 1024],
    [1920, 512],
    [1600, 640],
    [1152, 896],
    [1792, 576],
    [1472, 704],
    [960, 1088],
    [1472, 704],
    [1344, 768],
    [1344, 768],
    [1024, 960],
    [1024, 960],
]

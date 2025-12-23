import React, {useCallback, useEffect, useRef, useState} from 'react';
import {marked} from 'marked';
import DOMPurify from 'dompurify';
// @ts-ignore
import katex from 'katex';

// --- Constants ---
const A4_WIDTH_PT = 595; // Standard A4 width in points (72dpi)
const A4_HEIGHT_PT = 842; // Standard A4 height in points (72dpi)
const PAGE_PADDING_PT = 40; // 40pt padding
const COLUMN_GAP_PT = 10; // Space between columns (pt)

const MAX_FONT_SIZE = 16;
const MIN_FONT_SIZE = 2;
const FONT_STEP = 0.1;

const PT_TO_PX = 96 / 72; // CSS 1pt = 1/72in, 1px = 1/96in
const ptToPx = (pt: number) => pt * PT_TO_PX;

const PAGE_PADDING_PX = ptToPx(PAGE_PADDING_PT);
const COLUMN_GAP_PX = ptToPx(COLUMN_GAP_PT);

// Helper function to get page dimensions based on orientation
const getPageDimensions = (orientation: 'portrait' | 'landscape') => {
    const pageWidthPt = orientation === 'portrait' ? A4_WIDTH_PT : A4_HEIGHT_PT;
    const pageHeightPt = orientation === 'portrait' ? A4_HEIGHT_PT : A4_WIDTH_PT;
    const columnHeightPt = pageHeightPt - (PAGE_PADDING_PT * 2);

    return {
        pageWidthPt,
        pageHeightPt,
        columnHeightPt,
        pageWidthPx: ptToPx(pageWidthPt),
        pageHeightPx: ptToPx(pageHeightPt),
        columnHeightPx: ptToPx(columnHeightPt)
    };
};

const DEFAULT_MARKDOWN = `
### THE EXAMPLE CHEATSHEET
***AI vs. Computational Intelligence (CI)***

**AI (Symbolic):** Top-down, logical, rule-based, manipulation of symbols. Good for precise problems (e.g., chess).

**CI (Sub-symbolic):** Bottom-up, numeric, learning from data, bioinspired. Includes Neural Networks, Fuzzy Systems, Evolutionary Computation.

---

***Neural Networks*** Formulation: $ANN = (ARCH, RULE)$, $ARCH$: architecture refers to the combination of components, $RULE$: refers to the set of rules that relate the components.

$ARCH = (u, v, w, x, y)$, simple and alike neurons represented by $u$ and $v$ in N-dimensional space; inter-neuron connection weights represented by $w$ in M-dimensional space; external input and outputs represented by $x$ and $y$ in n and m-dimensional space. $RULE = (E, F, G, H, L)$, $E$: evaluation rule mapped from $v$ and/or $y$ to a real line, $F$: activation rule mapped from $u$ to $v$, $G$: aggregation rule mapped from $v$, $w$, and/or $x$ to $u$, $H$: output rule mapped from $v$ to $y$, $y$ usually a subset of $v$, $L$: learning rule mapped from $v$, $w$, and $x$ to $w$, usually iterative.

**Activation Functions:** 1.Sigmoid: $\\varphi(v) = \\frac{1}{1 + e^{-av}} \\in (0, 1)$; 2.Tanh: $\\varphi(v) = \\tanh(v) \\in (-1, 1)$; 3.ReLU: $\\varphi(v) = \\max(0, v)$.

---

***Neural Networks Comparing*** 

$F$: **Determinstic vs. Stochastic**

**Determinstic**: Same input always yields the same output. Represent point values. Generally lower complexity; easier to train. Does not naturally express "doubt."

**Stochastic**: Same input can yield different outputs. Represent distributions. Higher complexity; often requires complex sampling. Can quantify how "unsure" it is. 

$G$ & $H$: **Feedforward vs. Recurrent**

**Feedforward**: Information moves in one direction only—from input to output. No loops or cycles. Does not have "memory" of previous inputs; treats each piece of data independently. Ideal for static data like image classification.

**Recurrent**: Contains feedback loops that allow information to persist. Maintains an internal "hidden state" that acts as memory. Designed for sequential or time-series data where context matters. Ideal for speech recognition and language translation (e.g., LSTMs, RNNs).

$G$: **Semilinear vs. Higher-Order**

**Semilinear**: The most common type (e.g., standard MLPs). The net input to a neuron is a linear combination of inputs ($w \\cdot x$), which is then passed through a non-linear activation function. Computationally efficient but relies on the activation function to capture all non-linearities.

**Higher-Order**: Uses products or "conjuncts" of inputs (e.g., $x_1 \\cdot x_2$) as part of the net input. Can capture complex correlations between features directly in the input layer. Higher learning capacity for specific patterns but suffers from the "curse of dimensionality" as the number of weight connections explodes.

$L$: **Supervised vs. Unsupervised**

**Supervised**: Models are trained using labeled datasets (input-output pairs). The goal is to learn a mapping function to predict outcomes for new data. High accuracy on specific tasks but requires expensive, human-labeled data. Common for classification and regression.

**Unsupervised**: Models work with unlabeled data. The goal is to find hidden structures, patterns, or groupings within the raw data itself without external guidance. Useful for discovering new insights but results can be harder to validate. Common for clustering (K-Means) and dimensionality reduction (PCA).

---

***McCulloch-Pitts Neuron & Threshold Logic***

**Model:** $x_i \\in \\{0, 1\\}$ or $\\{-1, 1\\}$; $u = sum_i w_i x_i - \\theta$, output $y = step(u)$. Interpretation of parameters: $w_i \\gt 0$ means excitatory connection; $w_i \\lt 0$ inhibitory. $\\theta$ controls global threshold.

**Threshold logic:** a single neuron implements a linear threshold function, a multi-layer network of such units can represent any Boolean function.

**Linear separability:** Two pattern sets are linearly separable if there exists $(w, \\theta)$ such that $w^T x \\gt 0$ for class $+1$ and $w^T x \\lt 0$ for class $-1$. Geometrically, classes are separable by a hyperplane; algebraically, constraints can be written as a system of linear inequalities.

**Importance:** determines which problems a single-layer perceptron can solve. XOR and parity are classic examples of non-linear separable tasks.

---

***Perceptron & ADALINE***

**Single-layer Learning Machines:** Perceptron architecture is a single adaptive neuron with hard-limit activation. Decision rule $y = \\text{sign}(w^T - \\theta)$, $y \\in {+1, -1}$. Learning rule (online): for each sample $(x_p, t_p)$ with $t_p \\in \\{+1, -1\\}$, if misclassified, update $w < -w + \\eta * t_p * x_p$, $\\theta < -\\theta - \\eta * t_p$, if correctly classified, no change. Intuition: move decision boundary towards misclassified point so that it becomes correctly classified next time.

**Perceptron Convergence Theorem:** If the data are linearly separable and learning rate eta is small but constant, the perceptron will find some separating hyperplane in a finite number of updates.

**ADALINE:** Uses linear activation $y = w^T x$; error $e_p  = t_p  - y_p$. Cost function: $E = 1/2 * sum_p e_p^2$ (mean-squared error). LMS update (online): $w \\leftarrow w + η * e_p  * x_p$, which is gradient descent on $E$. Because the model is linear and $E$ is quadratic in $w$, the error surface is convex and LMS converges to the unique global minimum (for small enough eta).

**Comparison:** Perceptron uses step activation and classification error; ADALINE uses linear activation and MSE; perceptron may not converge for non-separable data; ADALINE always converges to least-squares solution but may still misclassify.

**Perceptron Rule:** $w(t+1) = w(t) + \\eta (d - y)x$ (Uses hard limiter).

**LMS / Delta Rule (ADALINE):** Uses the derivative of linear error (Gradient Descent).

---

***Multi-Layer Perceptron (MLP)***
**Backpropagation:**
Output layer error: $\\delta_k = (d_k - y_k) \\varphi'(v_k)$; Hidden layer error: $\\delta_j = \\varphi'(v_j) \\sum_k \\delta_k w_{kj}$; Weight update: $\\Delta w_{ji} = \\eta \\delta_j y_i$

---

***Radial Basis Function (RBF)***
A feedforward neural network with a single hidden layer that uses radial basis functions as activation units. It performs a non-linear transformation in the hidden layer and a linear combination in the output layer.

(1) ArchitectureInput Layer: $N$ source nodes (passes input directly).Hidden Layer: $J$ hidden units. Each unit $j$ has:A Center vector $c_j$.A Width parameter $\\sigma_j$ (spread).A Non-linear Activation $\\varphi(\\cdot)$.Output Layer: Linear neurons that sum the weighted outputs of the hidden layer.

(2) Mathematical ModelThe output $y_k(x)$ for an input vector $x$ is:$$y_k(x) = \\sum_{j=1}^{J} w_{kj} \\varphi_j(x) + w_{k0}$$$\\varphi_j(x)$: Response of the $j$-th hidden neuron.$w_{kj}$: Weight connecting hidden node $j$ to output node $k$.$w_{k0}$: Bias term.Gaussian Basis Function (Most Common):The activation depends on the Euclidean distance between input $x$ and center $c_j$:$$\\varphi_j(x) = \\exp \\left( - \\frac{||x - c_j||^2}{2\\sigma_j^2} \\right)$$If $x$ is close to center $c_j$, output $\\approx 1$.If $x$ is far from $c_j$, output $\\rightarrow 0$.

(3) Cover’s Theorem"A complex pattern-classification problem cast in a high-dimensional space non-linearly is more likely to be linearly separable than in a low-dimensional space."Logic: RBF maps input space (non-linear) $\\rightarrow$ Hidden space (high-dim, linear).

(4) Learning Algorithms (Calculation Methods)Training is hybrid (2 stages), which makes it much faster than Backpropagation.

Step 1: Unsupervised Learning (Hidden Layer Parameters)We need to find the Centers ($c_j$) and Widths ($\\sigma_j$). Method A: Fixed Centers (Random)Randomly select $J$ points from the training data to be centers.$\\sigma$ is fixed to average distance between centers: $\\sigma = \\frac{d_{max}}{\\sqrt{2J}}$. Method B: Self-Organized Selection (k-means Clustering)Use k-means clustering to find $J$ cluster centroids.Set $c_j = \\text{centroid of cluster } j$.Set width $\\sigma_j$ based on the variance of data in that cluster (or distance to nearest neighbor). **Formula (P-Nearest Neighbor):** $\\sigma_j = \\sqrt{\\frac{1}{P} \\sum_{p=1}^{P} ||c_j - c_p||^2}$

Step 2: Supervised Learning (Output Weights) Once $\\varphi(x)$ is fixed, the system becomes a simple Linear Equation: $D = \\Phi W$. We want to minimize the error: $E = \\frac{1}{2} \\sum (d_k - y_k)^2$. Exact Solution (Pseudo-Inverse): $W = (\\Phi^T \\Phi)^{-1} \\Phi^T D$, $\\Phi$: Matrix of hidden layer outputs for all data points (Dimensions: $N_{samples} \\times J_{hidden}$). $D$: Target vector. Iterative Solution (LMS / Gradient Descent): If the matrix is too large to invert, use the Delta Rule: $\\Delta w_{kj} = \\eta (d_k - y_k) \\varphi_j(x)$

---

***Support Vector Machines (SVM)***

**Goal:** Maximize the margin (distance to the nearest data points) to achieve maximal generalization power and minimize structural risk. (Proposed by Vapnik).

(1) Hard Margin (Linearly Separable Data)Primal Problem: Minimize the weight magnitude to maximize margin width.Objective: Minimize $J(w) = \\frac{1}{2} ||w||^2$ Subject to: $y_i (w^T x_i + b) \\ge 1$ for all $i=1 \\dots n$. Margin Width: $\\text{Margin} = \\frac{2}{||w||}$ Lagrangian Primal: $L_p(w, b, \\alpha) = \\frac{1}{2}||w||^2 - \\sum_{i=1}^{n} \\alpha_i [y_i(w^T x_i + b) - 1]$ Dual Problem (The one we actually solve): Maximize: $L_D(\\alpha) = \\sum_{i=1}^{n} \\alpha_i - \\frac{1}{2} \\sum_{i=1}^{n} \\sum_{j=1}^{n} \\alpha_i \\alpha_j y_i y_j (x_i^T x_j)$ Subject to: $\\sum_{i=1}^{n} \\alpha_i y_i = 0$ AND $\\alpha_i \\ge 0$.

(2) Soft Margin (Non-Linearly Separable / Noisy Data) Slack Variables ($\\xi_i$): Allow some points to violate the margin. New Objective: Minimize $\\frac{1}{2} ||w||^2 + C \\sum_{i=1}^{n} \\xi_i$$C$ (Penalty Parameter): Large $C$: Hard margin behavior (strict, potential overfitting). Small $C$: Soft margin (allows errors, smoother boundary).Dual Constraints Change: $0 \\le \\alpha_i \\le C$ (The only difference in calculation!).

(3) The Kernel TrickConcept: Map input $x$ to a high-dimensional feature space $\\phi(x)$ where it becomes linearly separable. Implementation: Replace the dot product $x_i^T x_j$ in the Dual Problem with a Kernel function $K(x_i, x_j)$. $K(x_i, x_j) = \\phi(x_i)^T \\phi(x_j)$ Common Kernels (Memorize these formulas): Linear: $K(x_i, x_j) = x_i^T x_j$ Polynomial: $K(x_i, x_j) = (1 + x_i^T x_j)^p$ Gaussian (RBF): $K(x_i, x_j) = \\exp \\left( - \\frac{||x_i - x_j||^2}{2\\sigma^2} \\right)$ Sigmoid: $K(x_i, x_j) = \\tanh(\\beta_0 x_i^T x_j + \\beta_1)$.

(4) Key Definitions & TheoremsSupport Vectors: The data points where $\\alpha_i > 0$. Only these points determine the decision boundary. Removing other points changes nothing.Discriminant Function: Final classifier equation.$g(x) = \\text{sign} \\left( \\sum_{i \\in SV} \\alpha_i y_i K(x_i, x) + b \\right)$ Mercer’s Theorem: A function $K(x, y)$ is a valid kernel if and only if its Gram matrix is semi-positive definite symmetric. KKT Conditions: $\\alpha_i [y_i(w^T x_i + b) - 1] = 0$. This implies that for any support vector (where $\\alpha_i \\gt 0$), the constraint must be exactly 1 (it lies on the margin).

---

***Boltzmann Machine (BM)***
Boltzmann Machine is a type of stochastic recurrent neural network. It is rooted in statistical mechanics, specifically the Boltzmann distribution (Gibbs distribution), which describes the probability of a system being in a certain state based on its energy and temperature.

**Type:** Unsupervised, Generative Model.
It defines a probability distribution over binary patterns. The network seeks to reach a state of "thermal equilibrium" where the energy of the system is minimized globally.

**Stochastic Nature:** Unlike deterministic networks (like a standard perceptron), neurons in a BM switch on (1) or off (0) probabilistically based on the weighted input they receive and the system's current temperature.

**Architecture:** A network of symmetrically connected units (neurons): Visible Units ($v$): The interface with the environment (where data is input or read out). Hidden Units ($h$): Latent variables that capture complex dependencies and higher-order correlations in the data.

**Connections:** Fully Recurrent: Every node is connected to every other node (in a general BM). Symmetric Weights: The weight $w_{ij}$ from unit $i$ to $j$ is the same as from $j$ to $i$ ($w_{ij} = w_{ji}$). No Self-Connections: Typically $w_{ii} = 0$.

**Calculations:** $v_i = \\{-1, 1\\}^n$, $u_i = \\sum_{j}w_{ij}v_j + x_i$, $w_{ij} = w_{ji}$, $E(v) = -\\frac{1}{2}\\sum_{i}\\sum_{j \\neq i}w_{ij}v_i v_j - \\sum_{i}x_i v_i$, $\\Delta E(v_i) := E(-v_i) = \\frac{\\partial{E}}{\\partial{v_i}}\\Delta v_i = (\\sum_{j} w_{ij}w_j + x_i) 2 v_i = 2 u_i v_i$, $P(-v_i \\rightarrow v_i) = \\frac{1}{1 + \\text{exp}(\\frac{\\Delta E(v_i)}{T})} = \\frac{1}{1 + \\text{exp}(-\\frac{2 u_i v_i}{T})}$, $P(-v_i = -1 \\rightarrow 1) = \\frac{1}{1 + \\text{exp}(\\frac{2 u_i}{T})}$, $P(-v_i = 1 \\rightarrow -1) = \\frac{1}{1 + \\text{exp}(\\frac{2 u_i}{T})}$

**Hopfield Networks / Boltzmann Machines:** Energy Function: $E = -\\frac{1}{2} \\sum \\sum w_{ij} s_i s_j$. (Crucial for understanding stability). Hebbian Learning Rule: $\\Delta w_{ij} = \\eta (x_i x_j)$.

**Energy Function:** $E = -\\frac{1}{2} \\sum \\sum w_{ij} s_i s_j$. (Crucial for understanding stability).

**Hebbian Learning Rule:** $\\Delta w_{ij} = \\eta (x_i x_j)$.

---

***Mean Field Annealing Network***
A deterministric reccurent neural network. Based on mean-field theory. Continuous state variables on $[-1, 1]^n$. Use a bipolar sigmoid activation function. Use a gradually decreasing temperature parameter like simulated annealing. Used for combinatorial optimization.

**Calculation:** $u_i = \\sum_j w_j v_j + x_i$, $v_i = \\frac{1 - \\text{exp}(-\\frac{2 u_i}{T})}{1 + \\text{exp}(-\\frac{2 u_i}{T})} = \\tanh(\\frac{u_i}{T})$, $T \\geq 0$, $\\frac{dT}{dt} < 0$. $\\lim_{t \\rightarrow \\infty}{T} = 0$. As $T \\rightarrow 0$, $v_i \\in \\{-1, 1\\}$. $E(v) = -\\frac{1}{2} \\sum_i\\sum_j w_{ij} v_i v_j - \\sum_i x_i v_i$, $v_i \\in [-1, 1]$. $\\text{Exp}(v_i) = P(v_i = 1) - P(v_i = -1) = P(v_i = 1) - [1 - P(v_i = 1)] = 2 P(v_i = 1) - 1 = \\frac{2}{1 + \\text{exp}(-\\frac{2 u_i}{T})} - 1 = \\frac{1 - \\text{exp}(-\\frac{2 u_i}{T})}{1 + \\text{exp}(-\\frac{2 u_i}{T})} = \\tanh(\\frac{u_i}{T})$. $v_i = \\tanh(-\\frac{1}{T} \\frac{\\partial{E(v)}}{\\partial{v_i}})$.

---

***Self-Organizing Maps (SOM)***
A single-layer network with a winner-take-all layer using an unsupervised learning algorithm. Formation of a topographic map through self-organization. Map high-dimensional data to one or two-dimensional feature maps.

**Kohonen's Learning Algorithm:**
(1) Initialization: randomize $w_{ij}(0)$ for $i = 1, 2, 3, ..., n$, $j = 1, 2, ..., m$, $p = 1$, $t = 0$.

(2) Distance: for datum $x^p$, $d_j = \\sum^n[x_i^p - w_{ij}(t)]^2$.

(3) Minimize: Find $k$ such that  $d_k = \\min_j d_j$.

(4) Adaptation: $\\forall j \\in N_k(t), i = 1, 2, ..., n$, $\\Delta w_{ij}(t) = \\eta(t)[x_i^p - w_{ij}(t)]$. $0 \\leq \\eta \\lt 1$, $p \\leftarrow p + 1$, go to (2)Distance.

SOM is presented as a method for Unsupervised Learning and Spatial Mapping.

---

***Echo State Network (ESN)*** Also called reservoir computiing. A recurrent neural network with sparse connections and random weights among hidden neurons.
ESN is presented immediately after as a method for Recurrent Learning and Temporal Processing.

---

***Fuzzy Logic*** A generalization of classical logic, describes one kind of uncertainty: **imprreciseness** or **ambiguity**. Probably on the other hand describes the other kind of uncertainty: **randomness**.

**Membership Function:** Let $X$ be a classical set. A membership function of fuzzy set $A$: $u_A : X \\rightarrow [0, 1]$ defines the fuzzy set $A$ of $X$. Crips sets are special case of fuzzy sets where the values of the membership function are 0 and 1 only.

**Fuzzy Set:** Fuzzy set $A$ is the set of all pairs $(x, u_A(x))$ where $x$ belongs to $X$, i.e., $A = \\{(x, u_A(x)) | x \\in X\\}$, if $X$ is discrete, $A = \\sum_i u_A(x_i) / x_i$; if $X$ is continuous, $A = \\int_X u_A(x) / x$. Support set of $A$ is $\\text{supp}(A) = \\{x \\in X | u_A(x) > 0\\}$.

**Fuzzy Set Terminology:** Normalized fuzzy set $A$: its height is unity, i.e., $ht(A) = 1$; otherwise: it is subnormal. $\\alpha$-cut of a fuzzy set $A$: a crisp set $A_\\alpha = \\{x \\in X | u_A(x) \\geq \\alpha\\}$. Convex fuzzy set $A$: $\\forall \\lambda \\in [0, 1]$, $\\forall x, y \\in X$, $u_A(\\lambda x + (1 - \\lambda) y) \\geq \\min(u_A(x), u_A(y))$, i.e., any $\\alpha$-cut is a convex set.

**Logic Operations on Fuzzy Sets:** Union of two fuzzy sets: $\\mu_{A \\cup B}(x) = \\max\\{\\mu_A(x), \\mu_B(x)\\}$; Intersection of two fuzzy sets: $\\mu_{A \\cap B}(x) = \\min\\{\\mu_A(x), \\mu_B(x)\\}$; Complement of a fuzzy set: $\\mu_{\\bar A}(x) = 1 - \\mu_A(x)$. Equality: for all $x$, $u_A(x) = u_B(x)$. Degree of equality: $E(A, B) = \\text{deg}(A = B) = \\frac{|A \\cap B|}{|A \\cup B|}$.

**Properties of Fuzzy Sets:** Union: $A \\subseteq A \\cup B$; Intersection: $A \\cap B \\subseteq A$, $A \\cap B \\subseteq B$; Double negation law: $A = A$; DeMorgan's Law: $\\overline{A \\cup B} = \\overline{A} \\cap \\overline{B}$, $\\overline{A \\cap B} = \\overline{A} \\cup \\overline{B}$; However, $A \\cup \\overline{A} \\neq X$, $A \\cap \\overline{A} \\neq \\emptyset$.

**Cardinality and Entropy of Fuzzy Sets** Cardinality: $|A|$ is defined as the sum of the membership function values of all elements in $X$, i.e., $|A| = \\sum_{x \\in X} \\mu_A(x)$ or $|A| = \\int_X \\mu_A(x)dx$. Entropy: $E(A) = \\frac{|A \\cap \\overline{A}|}{|A \\cup \\overline{A}|}$

**Fuzzy Relations:** $R(x_1, x_2, ..., x_n) = \\int_{X_1, X_2, ..., X_n}u_R(x_1, x_2, ..., x_n) / (x_1, x_2, ..., x_n)$, $R(x_1, x_2, ..., x_n) = \\{((x_1, x_2, ..., x_n), u_R(x_1, x_2, ..., x_n))|(x_1, x_2, ..., x_n) \\in X_1 \\times X_2 ... X_n\\}$.

---

***Fuzzifiers & Defuzzifiers*** Fuzzifier is a mapping from a real-valued set to a fuzzy by means of a membership function. Defuzzifiers is a mapping from a fuzzy set to a real-valued set.

**Typical Defuzzifiers:** Centroid: $x^* = \\frac{\\int_X x \\mu(x)dx}{\\int_X \\mu(x)dx}$ or $x^* = \\frac{\\sum_i x_i \\mu_A(x_i)}{\\sum_i \\mu_A(x_i)}$. Center average defuzzifier: $x^* = \\frac{\\sum_i x_i ht(x_i)}{\\sum_i ht(x_i)}$.

**Linguistic Variables:**

1. Important to fuzzy logic and approximate reasoning;
2. Variables whose values are words or sentences in natural or artificial languages.

**Mamdani-type Fuzzy Systems:** a type of fuzzy inference system that uses linguistic IF-THEN rules to map inputs to outputs, originally developed to mimic human control systems. The key characteristic: output of each rule is a fuzzy set, which is then aggregated and defuzzified to produce a final, crips output value.

**Fuzzy Inference Process:**
1. when imprecise information is input to a fuzzy inference system, first quantified by constructing a membership function;
2. based on a fuzzy rule base, fuzzy inference engine makes a fuzzy decision;
3. fuzzy decision is then defuzzified to output for an action;
4. defuzzification is usually done by using centroid method.

---

***Takagi-Sugeno Fuzzy Systems (T-S)*** a type of fuzzy inference system that uses linear function in its rule consequents to model complex, nonlinear systems. Key characteristic: output of each rule is a fuzzy set, then aggregated.

**Fuzzy Modeling:** A nonlinear system is approximated by a set of fuzzy "IF-THEN" rules. Rule structure: Each rule has a premise (IF) and a consequence (THEN). Linear consequences: The consequent is a linear equation of the input variables and a constant.

**T & S** describe a fuzzy implication $R$ as: $R$: if ($x_1$ is $\\mu_A(x_1)$, ... $x_k$ is $\\mu_A(x_k)$) then $y = g(x_1, x_2, ..., x_k)$. **Output Calculation:** The final output is calculated as a weightd average of the outputs of all the rules. The "weight" for each rule is determined by how well the current system state matches the premise of that rule.

* *Mamdani:* Output is a fuzzy set.
* *T-S:* Output is a linear function/constant.

---

***Type-2 Fuzzy Logic*** a generalization of type-1 fuzzy logic to handle uncertainty of membership function by using fuzzy membership function.

---

***Evolutionary Computation & Optimization***

**Genetic Algorithms (GA):**
Definition: A stochastic search method simulating natural evolution (John Holland, 1970s). Useful for discontinuous/non-differentiable fitness functions.

**Key Components:Population:** Set of candidate solutions (chromosomes/strings). Fitness Function: Evaluates how good a solution is. Selection (Reproduction): "Survival of the fittest." Roulette Wheel Selection: Probability of selection is proportional to fitness. Method: Generate random $n \\in [0, \\text{TotalFitness}]$. Pick first member where running sum $\\ge n$. Crossover (Recombination): Exchanges parts of two parent strings to create offspring. Occurs with probability $P_c$ (usually high).Mutation: Randomly flips bits to introduce new genetic information. Prevents premature convergence. Occurs with probability $P_m$ (usually low).

**Particle Swarm Optimization (PSO)** Definition: Stochastic optimization based on social behavior of bird flocks/fish schools (Kennedy & Eberhart, 1995). Mechanism: Agents (particles) fly through search space, adjusting velocity based on their own history and the swarm's best known position.Update Equations: For each particle $k$ at time $t$: Velocity Update: $v_k(t+1) = \\alpha v_k(t) + \\phi_1 r_1 (p_k - x_k(t)) + \\phi_2 r_2 (p_g - x_k(t))$. $\\alpha$: Inertia weight (momentum). $p_k$: Personal best position of particle $k$. $p_g$: Global best position of the entire swarm. $\\phi_1, \\phi_2$: Acceleration coefficients (Cognitive vs. Social). $r_1, r_2$: Random numbers $[0,1]$. Position Update: $x_k(t+1) = x_k(t) + v_k(t+1)$

**Ant Colony Optimization (ACO):** Definition: Probabilistic technique for finding paths through graphs (e.g., TSP), inspired by ants foraging (Marco Dorigo, 1992). Core Concept: Stigmergy (Indirect communication via environment). Ants lay pheromones on paths.Shorter paths get traversed faster $\\rightarrow$ more ants pass $\\rightarrow$ more pheromone accumulates.

Pheromone evaporates over time (preventing convergence to suboptimal loops). Process: Ants choose the next node probabilistically based on pheromone strength $\\tau$ and heuristic visibility $\\eta$.

**Simulated Annealing (SA)** Definition: Iterative search method inspired by the annealing process in metallurgy (heating and slow cooling). Mechanism: Accepts worse solutions with a probability $P$ to escape local minima.Metropolis Criterion: $P(\\text{accept}) = \\exp\\left(\\frac{-\\Delta E}{T}\\right)$ Temperature ($T$): Starts high (high randomness) and decreases (cooling schedule). As $T \\to 0$, SA behaves like simple Hill Climbing (Greedy).Connection: The Boltzmann Machine is a neural network implementation of Simulated Annealing.

`;

// --- LaTeX Rendering Helper ---
const renderLatex = (
    latex: string,
    displayMode: boolean
): string => {
    try {
        return katex.renderToString(latex, {
            displayMode,
            throwOnError: false,
            output: 'html',
            strict: false
        });
    } catch (e) {
        console.error('KaTeX rendering error:', e);
        return `<span class="katex-error" style="color: red;">${displayMode ? '$$' : '$'}${latex}${displayMode ? '$$' : '$'}</span>`;
    }
};

// Process LaTeX in text: handles both $...$ (inline) and $$...$$ (block)
const processLatex = (text: string): string => {
    // First, handle block-level LaTeX ($$...$$)
    // Using a placeholder approach to avoid nested replacements
    const blockPlaceholders: string[] = [];
    let processed = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, latex) => {
        const rendered = renderLatex(latex.trim(), true);
        const placeholder = `%%BLOCK_LATEX_${blockPlaceholders.length}%%`;
        blockPlaceholders.push(`<div class="katex-display">${rendered}</div>`);
        return placeholder;
    });

    // Then handle inline LaTeX ($...$), but not escaped \$ or already processed
    const inlinePlaceholders: string[] = [];
    processed = processed.replace(/(?<!\\)\$([^\$\n]+?)\$/g, (_, latex) => {
        const rendered = renderLatex(latex.trim(), false);
        const placeholder = `%%INLINE_LATEX_${inlinePlaceholders.length}%%`;
        inlinePlaceholders.push(rendered);
        return placeholder;
    });

    // Restore block placeholders
    blockPlaceholders.forEach((html, i) => {
        processed = processed.replace(`%%BLOCK_LATEX_${i}%%`, html);
    });

    // Restore inline placeholders
    inlinePlaceholders.forEach((html, i) => {
        processed = processed.replace(`%%INLINE_LATEX_${i}%%`, html);
    });

    return processed;
};

// Configure DOMPurify to allow KaTeX elements
const configureDOMPurify = () => {
    // Add KaTeX-specific tags and attributes to whitelist
    DOMPurify.addHook(
        'uponSanitizeElement',
        (node, data) => {
            if (data.tagName === 'annotation') {
                // Allow annotation elements used by KaTeX
                return;
            }
        }
    );

    // Allow SVG and MathML elements that KaTeX may use
    const ALLOWED_TAGS = [
        'math', 'semantics', 'mrow', 'mi', 'mo', 'mn', 'ms', 'mtext',
        'annotation', 'annotation-xml', 'mspace', 'mfrac', 'msqrt',
        'mroot', 'msub', 'msup', 'msubsup', 'munder', 'mover', 'munderover',
        'mtable', 'mtr', 'mtd', 'mlabeledtr', 'mmultiscripts', 'mprescripts',
        'none', 'menclose', 'mstyle', 'mpadded', 'mphantom', 'mglyph',
        'svg', 'line', 'path', 'g', 'rect', 'circle', 'ellipse', 'polygon',
        'polyline', 'text', 'tspan', 'image', 'use', 'defs', 'clipPath',
        'mask', 'pattern', 'marker', 'linearGradient', 'radialGradient', 'stop'
    ];

    const ALLOWED_ATTR = [
        'class', 'style', 'href', 'xmlns', 'mathvariant', 'encoding',
        'displaystyle', 'scriptlevel', 'lspace', 'rspace', 'stretchy',
        'symmetric', 'maxsize', 'minsize', 'largeop', 'movablelimits',
        'accent', 'accentunder', 'linebreak', 'lineleading', 'linebreakstyle',
        'linebreakmultchar', 'indentalign', 'indentshift', 'indenttarget',
        'indentalignfirst', 'indentshiftfirst', 'indentalignlast', 'indentshiftlast',
        'depth', 'height', 'width', 'rowalign', 'columnalign', 'columnwidth',
        'groupalign', 'alignmentscope', 'rowspacing', 'columnspacing', 'rowlines',
        'columnlines', 'frame', 'framespacing', 'equalrows', 'equalcolumns',
        'side', 'minlabelspacing', 'rowspan', 'columnspan', 'data-mml-node',
        'd', 'fill', 'stroke', 'stroke-width', 'viewBox', 'preserveAspectRatio',
        'x', 'y', 'x1', 'x2', 'y1', 'y2', 'cx', 'cy', 'r', 'rx', 'ry',
        'transform', 'opacity', 'font-size', 'text-anchor', 'dominant-baseline'
    ];

    return {ALLOWED_TAGS, ALLOWED_ATTR};
};

const domPurifyConfig = configureDOMPurify();

// --- SVG Icons ---
const CheckCircleIcon: React.FC<{ className?: string }> = ({className}) => (
    <svg
        className={className}
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
    >
        <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
        />
    </svg>
);

const ExclamationTriangleIcon: React.FC<{ className?: string }> = ({className}) => (
    <svg className={className} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
    </svg>
);


const App: React.FC = () => {
    const [markdown, setMarkdown] = useState<string>(DEFAULT_MARKDOWN);
    const [formattedHtml, setFormattedHtml] = useState<string>('');
    const [finalFontSize, setFinalFontSize] = useState<number | null>(null);
    const [finalNumColumns, setFinalNumColumns] = useState<number>(2);
    const [totalPages, setTotalPages] = useState<number>(2);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
    const [columnOption, setColumnOption] = useState<'auto' | '1' | '2' | '3' | '4'>('auto');
    const [pageOrientation, setPageOrientation] = useState<'portrait' | 'landscape'>('portrait');
    const [previewWindow, setPreviewWindow] = useState<Window | null>(null);

    const measurementRef = useRef<HTMLDivElement>(null);

    // Calculate pages when content or settings change
    const calculatePages = useCallback(
        async (
            html: string,
            fontSize: number,
            numCols: number,
            dims: ReturnType<typeof getPageDimensions>
        ): Promise<number> => {
            if (!html || !fontSize || !measurementRef.current) {
                return 2;
            }

            const container = measurementRef.current;
            container.innerHTML = html;
            const currentColumnWidthPt = (dims.pageWidthPt - (PAGE_PADDING_PT * 2) - (COLUMN_GAP_PT * (numCols - 1))) / numCols;
            const currentColumnWidthPx = ptToPx(currentColumnWidthPt);

            container.style.width = `${currentColumnWidthPx}px`;
            container.style.fontSize = `${fontSize}pt`;

            await new Promise(resolve => requestAnimationFrame(resolve));

            const totalContentHeightPx = container.scrollHeight;
            const heightPerColumnPx = dims.columnHeightPx;

            const totalColumnUnits = Math.ceil(totalContentHeightPx / heightPerColumnPx);
            const numPages = Math.ceil(totalColumnUnits / numCols);

            container.innerHTML = '';
            return numPages > 0 ? numPages : 1;
        }, []
    );

    const handleFormat = useCallback(async () => {
        setIsLoading(true);
        setFormattedHtml('');
        setFinalFontSize(null);
        setStatusMessage(null);

        await new Promise(resolve => setTimeout(resolve, 50));

        if (!measurementRef.current) {
            setStatusMessage({type: 'error', text: 'Cannot initialize typesetting engine.'});
            setIsLoading(false);
            return;
        }

        const container = measurementRef.current;
        let foundFit = false;

        // Process LaTeX before Markdown parsing
        const markdownWithLatex = processLatex(markdown);
        const unsafeHtml = await marked.parse(markdownWithLatex);
        // Use extended config to allow KaTeX elements
        const cleanHtml = DOMPurify.sanitize(unsafeHtml, {
            ADD_TAGS: domPurifyConfig.ALLOWED_TAGS,
            ADD_ATTR: domPurifyConfig.ALLOWED_ATTR,
            ALLOW_DATA_ATTR: true
        });
        container.innerHTML = cleanHtml;

        // Get current dimensions based on orientation
        const dims = getPageDimensions(pageOrientation);

        // Determine columns to try: auto (4 -> 1) or user-selected fixed column count
        const columnsToTry = columnOption === 'auto' ? [4, 3, 2, 1] : [parseInt(columnOption, 10)];
        for (const numCols of columnsToTry) {
            const currentColumnWidthPt = (dims.pageWidthPt - (PAGE_PADDING_PT * 2) - (COLUMN_GAP_PT * (numCols - 1))) / numCols;
            // Threshold height value of 2 pages (px)
            const totalAvailableHeightPx = dims.columnHeightPx * 2 * numCols;
            container.style.width = `${ptToPx(currentColumnWidthPt)}px`;

            // Inner loop for font size (MAX -> MIN)
            for (let fontSize = MAX_FONT_SIZE; fontSize >= MIN_FONT_SIZE; fontSize -= FONT_STEP) {
                container.style.fontSize = `${fontSize}pt`;

                await new Promise(resolve => requestAnimationFrame(resolve));

                if (container.scrollHeight <= totalAvailableHeightPx) {
                    setFormattedHtml(cleanHtml);
                    setFinalFontSize(fontSize);
                    setFinalNumColumns(numCols);
                    setStatusMessage({
                        type: 'success',
                        text: `Done successfully! The optimized layout is ${numCols} column(s) with a ${fontSize.toFixed(1)} pt fontsize.`
                    });
                    foundFit = true;
                    break; // Exit font size loop
                }
            }
            if (foundFit) {
                break; // Exit columns loop
            }
        }

        if (!foundFit) {
            const chosenCols = columnOption === 'auto' ? null : parseInt(columnOption, 10);
            setStatusMessage({
                type: 'error', text: chosenCols
                    ? `Too many contents. Cannot type into 2 pages even in the minimized font size and ${chosenCols} columns layouts.`
                    : 'Too many contents. Cannot type into 2 pages even in the minimized font size and 1 column layouts.'
            });
            setFormattedHtml('');
            setFinalFontSize(null);
            setFinalNumColumns(2);
        }

        container.innerHTML = '';
        setIsLoading(false);
    }, [markdown, columnOption, pageOrientation]);

    // When column option changes, only update column count; keep current font size.
    useEffect(() => {
        if (!formattedHtml) return;
        if (columnOption === 'auto') return; // Auto layout only when clicking "Auto Typesetting"
        const chosenCols = parseInt(columnOption, 10);
        if (Number.isFinite(chosenCols)) {
            setFinalNumColumns(chosenCols);
        }
    }, [columnOption, formattedHtml]);

    // Open preview in a new popup window
    const openPreviewWindow = useCallback(async () => {
        if (!formattedHtml || !finalFontSize) {
            setStatusMessage({type: 'error', text: 'Please click "Auto Typesetting" to generate and preview first.'});
            return;
        }

        // Close existing preview window if open
        if (previewWindow && !previewWindow.closed) {
            previewWindow.focus();
            return;
        }

        const dims = getPageDimensions(pageOrientation);
        const pages = await calculatePages(formattedHtml, finalFontSize, finalNumColumns, dims);
        const columnWidthPx = (dims.pageWidthPx - (PAGE_PADDING_PX * 2) - (COLUMN_GAP_PX * (finalNumColumns - 1))) / finalNumColumns;

        // Generate pages HTML
        const pagesHtml = Array.from({length: pages}).map((_, pageIndex) => {
            const columnsHtml = Array.from({length: finalNumColumns}).map((_, colIndex) => {
                const overallColumnIndex = pageIndex * finalNumColumns + colIndex;
                const translateY = -dims.columnHeightPx * overallColumnIndex;
                return `
                    <div style="width: ${columnWidthPx}px; height: ${dims.columnHeightPx}px; overflow: hidden;">
                        <div class="prose-styles" style="font-size: ${finalFontSize}pt; line-height: 1.6; transform: translateY(${translateY}px); width: ${columnWidthPx}px;">
                            ${formattedHtml}
                        </div>
                    </div>
                `;
            }).join('');

            return `
                <div class="preview-page" style="width: ${dims.pageWidthPx}px; height: ${dims.pageHeightPx}px; background: white; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); margin-bottom: 16px;">
                    <div style="height: 100%; display: flex; padding: ${PAGE_PADDING_PX}px; gap: ${COLUMN_GAP_PX}px;">
                        ${columnsHtml}
                    </div>
                </div>
            `;
        }).join('');

        const windowWidth = Math.min(dims.pageWidthPx + 100, screen.availWidth - 100);
        const windowHeight = Math.min(dims.pageHeightPx + 200, screen.availHeight - 100);

        const newWindow = window.open('', 'preview', `width=${windowWidth},height=${windowHeight},scrollbars=yes,resizable=yes`);
        if (!newWindow) {
            setStatusMessage({
                type: 'error',
                text: 'Cannot open window. Please make sure that your browser allows window pop-out.'
            });
            return;
        }

        setPreviewWindow(newWindow);

        const pageSize = pageOrientation === 'portrait' ? 'A4' : 'A4 landscape';

        newWindow.document.write(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>Preview - ${pages} page(s) / ${finalNumColumns} column(s)</title>
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" crossorigin="anonymous">
                <style>
                    * { box-sizing: border-box; margin: 0; padding: 0; }
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        background: #f5f5f5;
                        padding: 20px;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }
                    .toolbar {
                        position: sticky;
                        top: 0;
                        background: white;
                        padding: 12px 20px;
                        border-radius: 4px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        margin-bottom: 20px;
                        display: flex;
                        align-items: center;
                        gap: 16px;
                        flex-wrap: wrap;
                        z-index: 100;
                    }
                    .toolbar label { font-size: 14px; color: #666; }
                    .toolbar select, .toolbar button {
                        padding: 6px 12px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        font-size: 14px;
                        background: white;
                        cursor: pointer;
                    }
                    .toolbar select:focus, .toolbar button:focus {
                        outline: none;
                        border-color: #333;
                    }
                    .toolbar button:hover { background: #f0f0f0; }
                    .toolbar .btn-primary {
                        background: #000;
                        color: white;
                        border-color: #000;
                    }
                    .toolbar .btn-primary:hover { background: #333; }
                    .toolbar .font-controls {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }
                    .toolbar .font-btn {
                        width: 28px;
                        height: 28px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: bold;
                    }
                    .toolbar .font-size {
                        min-width: 60px;
                        text-align: center;
                        font-family: monospace;
                    }
                    .preview-container {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }
                    .prose-styles h1, .prose-styles h2, .prose-styles h3, .prose-styles h4, .prose-styles h5, .prose-styles h6 {
                        color: #000 !important;
                        font-weight: 700 !important;
                    }
                    .prose-styles p, .prose-styles li {
                        color: #000 !important;
                    }
                    .prose-styles a {
                        color: #000 !important;
                        text-decoration: underline !important;
                    }
                    .prose-styles blockquote {
                        border-left-color: rgba(0,0,0,0.25) !important;
                        color: rgba(0,0,0,0.85) !important;
                        border-left-width: 4px !important;
                        padding-left: 1em !important; 
                    }
                    .prose-styles code {
                        color: #000 !important;
                        background-color: #f8fafc !important;
                        padding: 0.1em 0.3em !important;
                        border-radius: 2px !important;
                    }
                    .prose-styles pre {
                        background-color: #f5f5f5 !important;
                        color: #000 !important;
                        padding: 1em !important;
                        border-radius: 2px !important;
                    }
                    .prose-styles .katex {
                        font-size: 1em !important;
                    }
                    .prose-styles .katex-display {
                        display: block !important;
                        margin: 0.5em 0 !important;
                        text-align: center !important;
                        overflow-x: auto !important;
                        overflow-y: hidden !important;
                    }
                    .prose-styles .katex-display > .katex {
                        display: inline-block !important;
                        text-align: initial !important;
                    }
                    .prose-styles .katex-error {
                        color: #cc0000 !important;
                        font-family: monospace !important;
                        white-space: pre-wrap !important;
                    }
                    @media print {
                        body {
                            background: white;
                            padding: 0;
                        }
                        .toolbar {
                            display: none !important;
                        }
                        .preview-page {
                            box-shadow: none !important;
                            margin-bottom: 0 !important;
                            page-break-after: always;
                        }
                        @page {
                            size: ${pageSize};
                            margin: 0;
                        }
                    }
                </style>
            </head>
            <body>
                <div class="toolbar">
                    <label>Paper Direction:</label>
                    <select id="orientation">
                        <option value="portrait" ${pageOrientation === 'portrait' ? 'selected' : ''}>Portrait</option>
                        <option value="landscape" ${pageOrientation === 'landscape' ? 'selected' : ''}>Landscape</option>
                    </select>
                    <label>Columns:</label>
                    <select id="columns">
                        <option value="1" ${finalNumColumns === 1 ? 'selected' : ''}>1 Column</option>
                        <option value="2" ${finalNumColumns === 2 ? 'selected' : ''}>2 Columns</option>
                        <option value="3" ${finalNumColumns === 3 ? 'selected' : ''}>3 Columns</option>
                        <option value="4" ${finalNumColumns === 4 ? 'selected' : ''}>4 Columns</option>
                    </select>
                    <div class="font-controls">
                        <label>Font Size:</label>
                        <button class="font-btn" id="fontMinus">−</button>
                        <span class="font-size" id="fontSizeDisplay">${finalFontSize.toFixed(1)}pt</span>
                        <button class="font-btn" id="fontPlus">+</button>
                    </div>
                    <button class="btn-primary" id="printBtn">Print / Export to PDF</button>
                </div>
                <div class="preview-container" id="previewContainer">
                    ${pagesHtml}
                </div>
                <script>
                    const EXPORT_DPI = 300;
                    const EXPORT_IMAGE_QUALITY = 0.92;
                    const PAGE_PADDING_PX = ${PAGE_PADDING_PX};
                    const COLUMN_GAP_PX = ${COLUMN_GAP_PX};
                    const MIN_FONT_SIZE = ${MIN_FONT_SIZE};
                    const MAX_FONT_SIZE = ${MAX_FONT_SIZE};
                    const FONT_STEP = ${FONT_STEP};
                    
                    let currentOrientation = '${pageOrientation}';
                    let currentColumns = ${finalNumColumns};
                    let currentFontSize = ${finalFontSize};
                    const formattedHtml = ${JSON.stringify(formattedHtml)};
                    
                    function getPageDimensions(orientation) {
                        const A4_WIDTH_PT = 595;
                        const A4_HEIGHT_PT = 842;
                        const PAGE_PADDING_PT = 40;
                        const PT_TO_PX = 96 / 72;
                        const pageWidthPt = orientation === 'portrait' ? A4_WIDTH_PT : A4_HEIGHT_PT;
                        const pageHeightPt = orientation === 'portrait' ? A4_HEIGHT_PT : A4_WIDTH_PT;
                        const columnHeightPt = pageHeightPt - (PAGE_PADDING_PT * 2);
                        return {
                            pageWidthPt,
                            pageHeightPt,
                            columnHeightPt,
                            pageWidthPx: pageWidthPt * PT_TO_PX,
                            pageHeightPx: pageHeightPt * PT_TO_PX,
                            columnHeightPx: columnHeightPt * PT_TO_PX
                        };
                    }
                    
                    function renderPreview() {
                        const dims = getPageDimensions(currentOrientation);
                        const columnWidthPx = (dims.pageWidthPx - (PAGE_PADDING_PX * 2) - (COLUMN_GAP_PX * (currentColumns - 1))) / currentColumns;
                        
                        // Calculate pages needed
                        const tempDiv = document.createElement('div');
                        tempDiv.style.cssText = 'position:absolute;visibility:hidden;width:' + columnWidthPx + 'px;font-size:' + currentFontSize + 'pt;line-height:1.6';
                        tempDiv.className = 'prose-styles';
                        tempDiv.innerHTML = formattedHtml;
                        document.body.appendChild(tempDiv);
                        const totalHeight = tempDiv.scrollHeight;
                        document.body.removeChild(tempDiv);
                        
                        const totalColumnUnits = Math.ceil(totalHeight / dims.columnHeightPx);
                        const numPages = Math.max(1, Math.ceil(totalColumnUnits / currentColumns));
                        
                        document.title = 'Previewing - ' + numPages + 'Pages / ' + currentColumns + 'Columns';
                        
                        let pagesHtml = '';
                        for (let pageIndex = 0; pageIndex < numPages; pageIndex++) {
                            let columnsHtml = '';
                            for (let colIndex = 0; colIndex < currentColumns; colIndex++) {
                                const overallColumnIndex = pageIndex * currentColumns + colIndex;
                                const translateY = -dims.columnHeightPx * overallColumnIndex;
                                columnsHtml += '<div style="width:' + columnWidthPx + 'px;height:' + dims.columnHeightPx + 'px;overflow:hidden;">' +
                                    '<div class="prose-styles" style="font-size:' + currentFontSize + 'pt;line-height:1.6;transform:translateY(' + translateY + 'px);width:' + columnWidthPx + 'px;">' +
                                    formattedHtml +
                                    '</div></div>';
                            }
                            pagesHtml += '<div class="preview-page" style="width:' + dims.pageWidthPx + 'px;height:' + dims.pageHeightPx + 'px;background:white;box-shadow:0 10px 15px -3px rgba(0,0,0,0.1);margin-bottom:16px;">' +
                                '<div style="height:100%;display:flex;padding:' + PAGE_PADDING_PX + 'px;gap:' + COLUMN_GAP_PX + 'px;">' +
                                columnsHtml +
                                '</div></div>';
                        }
                        document.getElementById('previewContainer').innerHTML = pagesHtml;
                    }
                    
                    document.getElementById('orientation').addEventListener('change', function(e) {
                        currentOrientation = e.target.value;
                        renderPreview();
                    });
                    
                    document.getElementById('columns').addEventListener('change', function(e) {
                        currentColumns = parseInt(e.target.value);
                        renderPreview();
                    });
                    
                    document.getElementById('fontMinus').addEventListener('click', function() {
                        if (currentFontSize > MIN_FONT_SIZE) {
                            currentFontSize = Math.max(MIN_FONT_SIZE, currentFontSize - FONT_STEP);
                            document.getElementById('fontSizeDisplay').textContent = currentFontSize.toFixed(1) + 'pt';
                            renderPreview();
                        }
                    });
                    
                    document.getElementById('fontPlus').addEventListener('click', function() {
                        if (currentFontSize < MAX_FONT_SIZE) {
                            currentFontSize = Math.min(MAX_FONT_SIZE, currentFontSize + FONT_STEP);
                            document.getElementById('fontSizeDisplay').textContent = currentFontSize.toFixed(1) + 'pt';
                            renderPreview();
                        }
                    });
                    
                    document.getElementById('printBtn').addEventListener('click', function() {
                        window.print();
                    });
                </script>
            </body>
            </html>
        `);
        newWindow.document.close();
    }, [formattedHtml, finalFontSize, finalNumColumns, pageOrientation, previewWindow, calculatePages]);

    return (
        <div className="bg-neutral-50 min-h-screen font-sans text-black">
            <style>
                {`
                @media print {
                    body { margin: 0; padding: 0; }
                    .no-print { display: none !important; }
                }
                .prose-styles h1, .prose-styles h2, .prose-styles h3, .prose-styles h4, .prose-styles h5, .prose-styles h6 { color: #000 !important; font-weight: 700 !important; }
                .prose-styles p, .prose-styles li { color: #000 !important; }
                .prose-styles a { color: #000 !important; text-decoration: underline !important; }
                .prose-styles blockquote { border-left-color: rgba(0,0,0,0.25) !important; color: rgba(0,0,0,0.85) !important; border-left-width: 4px !important; padding-left: 1em !important; }
                .prose-styles code { color: #000 !important; background-color: #f8fafc !important; padding: 0.1em 0.3em !important; border-radius: 2px !important; }
                .prose-styles pre { background-color: #f5f5f5 !important; color: #000 !important; padding: 1em !important; border-radius: 2px !important; }
                
                /* KaTeX styles */
                .prose-styles .katex { font-size: 1em !important; }
                .prose-styles .katex-display { display: block !important; margin: 0.5em 0 !important; text-align: center !important; overflow-x: auto !important; overflow-y: hidden !important; }
                .prose-styles .katex-display > .katex { display: inline-block !important; text-align: initial !important; }
                .prose-styles .katex-error { color: #cc0000 !important; font-family: monospace !important; white-space: pre-wrap !important; }
                .prose-styles .katex .base { display: inline-block !important; }
                .prose-styles .katex .strut { display: inline-block !important; }
                `}
            </style>

            <div ref={measurementRef} style={{
                position: 'absolute',
                visibility: 'hidden',
                top: '-9999px',
                left: '-9999px',
                lineHeight: '1.6'
            }} className="prose-styles"></div>

            <header className="bg-white shadow-sm p-4 no-print">
                <div className="container mx-auto flex justify-between items-center">
                    <h1 className="text-3xl font-bold text-black">CityCheatsheet: The One-key Cheatsheet Generator</h1>
                </div>
            </header>

            <main className="container mx-auto p-4 md:p-8 max-w-4xl">
                <div className="flex flex-col no-print">
                    <div className="bg-white rounded-sm shadow-md flex-grow flex flex-col">
                        <div
                            className="p-4 border-b border-neutral-200 flex items-center justify-between flex-wrap gap-2"
                        >
                            <h2 className="text-xl font-semibold text-black">Input Markdown Contents:</h2>
                            <div className="flex items-center gap-2 flex-wrap">
                                <span className="text-sm text-black/60">Paper Direction:</span>
                                <select
                                    value={pageOrientation}
                                    onChange={(e) => setPageOrientation(e.target.value as 'portrait' | 'landscape')}
                                    className="text-sm border border-neutral-300 rounded-sm px-2 py-1 bg-white focus:outline-none focus:ring-2 focus:ring-neutral-800 focus:border-neutral-800"
                                    aria-label="Choose your cheatsheet direction"
                                >
                                    <option value="portrait">Portrait</option>
                                    <option value="landscape">Landscape</option>
                                </select>
                                <span className="text-sm text-black/60 ml-2">Split Columns:</span>
                                <select
                                    value={columnOption}
                                    onChange={(e) => setColumnOption(e.target.value as 'auto' | '1' | '2' | '3' | '4')}
                                    className="text-sm border border-neutral-300 rounded-sm px-2 py-1 bg-white focus:outline-none focus:ring-2 focus:ring-neutral-800 focus:border-neutral-800"
                                    aria-label="Choose column numbers"
                                >
                                    <option value="auto">Auto</option>
                                    <option value="1">1 Column</option>
                                    <option value="2">2 Columns</option>
                                    <option value="3">3 Columns</option>
                                    <option value="4">4 Columns</option>
                                </select>
                                <button
                                    onClick={handleFormat}
                                    disabled={isLoading}
                                    className="ml-2 bg-black text-white font-bold py-2 px-6 rounded-sm hover:bg-neutral-900 focus:outline-none focus:ring-2 focus:ring-neutral-800 transition-colors duration-200 disabled:bg-neutral-400 disabled:cursor-not-allowed text-sm no-print"
                                >
                                    {isLoading ? 'Typesetting...' : 'Auto Typesetting'}
                                </button>
                                <button
                                    onClick={openPreviewWindow}
                                    disabled={!formattedHtml || isLoading}
                                    className="bg-neutral-700 text-white font-bold py-2 px-6 rounded-sm hover:bg-neutral-800 focus:outline-none focus:ring-2 focus:ring-neutral-600 transition-colors duration-200 disabled:bg-neutral-400 disabled:cursor-not-allowed text-sm no-print"
                                >
                                    Preview
                                </button>
                            </div>
                        </div>
                        <textarea
                            value={markdown}
                            onChange={(e) => setMarkdown(e.target.value)}
                            className="w-full h-full flex-grow p-4 border-0 resize-none focus:ring-0 text-sm leading-6"
                            placeholder="Input or paste you Markdown here"
                            style={{minHeight: '50vh'}}
                        />
                        <div
                            className="p-4 border-t border-neutral-200 flex items-center justify-between bg-neutral-50 rounded-b-sm">
                            {statusMessage ? (
                                <div className={`flex items-center text-sm text-black`}>
                                    {statusMessage.type === 'success' ?
                                        <CheckCircleIcon className="w-5 h-5 mr-2"/> :
                                        <ExclamationTriangleIcon className="w-5 h-5 mr-2"/>
                                    }
                                    <span>{statusMessage.text}</span>
                                </div>
                            ) : <div/> /* Placeholder to keep layout consistent */}
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default App;

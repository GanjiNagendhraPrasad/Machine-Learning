
<h2>1. Machine Learning</h2>
<p>
Machine Learning (ML) is a subset of Artificial Intelligence that allows systems
to learn patterns from data and make predictions or decisions without being
explicitly programmed.
</p>

<p>
Instead of writing rules manually, we provide data and the algorithm learns the
relationship between inputs and outputs.
</p>

<hr>

<h2>2. Types of Machine Learning</h2>
<ul>
  <li>Supervised Learning</li>
  <li>Unsupervised Learning</li>
  <li>Reinforcement Learning</li>
</ul>

<p>
This document focuses on <b>Supervised Learning</b>.
</p>

<hr>

<h2>3. Supervised Learning</h2>
<p>
In supervised learning, the dataset contains both:
</p>
<ul>
  <li><b>Input features (X)</b></li>
  <li><b>Output labels (Y)</b></li>
</ul>

<p>
The model learns by comparing its predictions with the correct output.
</p>

<h3>Example</h3>
<table border="1" cellpadding="6">
  <tr>
    <th>Experience (X)</th>
    <th>Salary (Y)</th>
  </tr>
  <tr>
    <td>1 year</td>
    <td>3 LPA</td>
  </tr>
  <tr>
    <td>5 years</td>
    <td>10 LPA</td>
  </tr>
</table>

<hr>

<h2>4. Types of Supervised Learning</h2>

<h3>4.1 Classification</h3>
<p>
Classification problems have <b>categorical outputs</b>.
</p>

<ul>
  <li>Email spam detection (Spam / Not Spam)</li>
  <li>Disease prediction (Positive / Negative)</li>
  <li>Image classification (Cat / Dog)</li>
</ul>

<p><b>Common Algorithms:</b></p>
<ul>
  <li>Logistic Regression</li>
  <li>K-Nearest Neighbors (KNN)</li>
  <li>Decision Tree</li>
  <li>Support Vector Machine (SVM)</li>
</ul>

<hr>

<h3>4.2 Regression</h3>
<p>
Regression problems have <b>continuous numeric outputs</b>.
</p>

<ul>
  <li>House price prediction</li>
  <li>Salary prediction</li>
  <li>Temperature prediction</li>
</ul>

<p><b>Common Algorithms:</b></p>
<ul>
  <li>Linear Regression</li>
  <li>Polynomial Regression</li>
  <li>Ridge and Lasso Regression</li>
</ul>

<hr>

<h2>5. Simple Linear Regression</h2>
<p>
Simple Linear Regression models the relationship between one input variable (X)
and one output variable (Y).
</p>

<h3>Mathematical Equation</h3>
<p>
<b>y = mx + c</b>
</p>

<ul>
  <li><b>m</b> – slope of the line</li>
  <li><b>c</b> – intercept</li>
</ul>

<p>
The goal is to find the best-fitting straight line that minimizes prediction error.
</p>

<hr>

<h2>6. Train–Test Split</h2>
<p>
To evaluate model performance fairly, the dataset is split into:
</p>

<ul>
  <li><b>Training data</b> – used to train the model</li>
  <li><b>Testing data</b> – used to evaluate the model</li>
</ul>

<p>
A common split ratio is <b>80% training</b> and <b>20% testing</b>.
</p>

<pre>
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
</pre>


<hr>

<h2>11. Workflow of Simple Linear Regression</h2>
<ol>
  <li>Collect data</li>
  <li>Split data into training and testing sets</li>
  <li>Train the model</li>
  <li>Minimize loss using gradient descent</li>
  <li>Make predictions</li>
  <li>Evaluate using MSE and R² score</li>
</ol>

<hr>



<hr>

<h2>1. Linear Regression</h2>
<p>
Linear Regression is a basic and commonly used predictive analysis technique in
Machine Learning and Statistics. It is used to model the relationship between
a dependent variable (Y) and one or more independent variables (X).
</p>

<p>
The relationship between the variables is assumed to be linear, meaning it can
be represented using a straight line.
</p>

<hr>

<h2>2. Simple Linear Regression</h2>
<p>
Simple Linear Regression deals with only one independent variable (X) and one
dependent variable (Y). It assumes that a linear relationship exists between the
response variable and the explanatory variable.
</p>

<p>
This relationship is modeled using a linear surface called a <b>hyperplane</b>.
</p>

<h3>Hyperplane Explanation</h3>
<p>
A hyperplane is a subspace that has one dimension less than the space it exists in.
In simple linear regression:
</p>

<ul>
  <li>One dimension represents the input variable (X)</li>
  <li>One dimension represents the output variable (Y)</li>
</ul>

<p>
Thus, the total dimensions are two, and a hyperplane in two dimensions is simply
a <b>straight line</b>.
</p>

<hr>

<h2>3. Uses of Regression Analysis</h2>

<h3>1. Measuring Relationship Strength</h3>
<p>
Regression is used to identify the strength of the effect that independent
variables have on a dependent variable.
</p>

<p><b>Examples:</b></p>
<ul>
  <li>Dose and effect</li>
  <li>Sales and marketing spending</li>
  <li>Age and income</li>
</ul>

<h3>2. Measuring Impact of Change</h3>
<p>
Regression analysis helps understand how much the dependent variable changes
when the independent variable changes.
</p>

<p>
Example question: How much additional sales income is generated for every
₹1000 spent on marketing?
</p>

<h3>3. Prediction of Future Values</h3>
<p>
Regression can be used to predict trends and future values.
</p>

<p><b>Examples:</b></p>
<ul>
  <li>Gold price after 6 months</li>
  <li>Future salary</li>
  <li>Trend estimation</li>
</ul>

<hr>

<h2>4. Line Equation in Linear Regression</h2>
<p>
The linear regression model is represented by the following equation:
</p>

<pre>
y = mx + c + e
</pre>

<ul>
  <li><b>y</b> – Actual output value</li>
  <li><b>x</b> – Input variable</li>
  <li><b>m</b> – Slope of the line</li>
  <li><b>c</b> – Intercept (value of y when x = 0)</li>
  <li><b>e</b> – Error (difference between actual and predicted value)</li>
</ul>

<p>
The goal of linear regression is to find values of <b>m</b> and <b>c</b> such that
the error is minimized.
</p>

<hr>

<h2>5. Predicted Value and Error</h2>
<p>
The predicted value is given by <b>mx + c</b>. The error represents the difference
between the predicted value and the actual value.
</p>

<p>
The model always tries to minimize this error.
</p>

<hr>

<h2>6. Cost Function</h2>
<p>
The error in a regression model is also called the <b>cost function</b>.
</p>

<p>
The cost function measures the total squared error over the training dataset.
</p>

<h3>Cost Function Formula</h3>
<pre>
Cost Function = (Predicted Value - Actual Value)²
</pre>

<pre>
Cost Function = (y[i] - (m * x[i] + c))²
</pre>

<p>
Squaring the error ensures all errors are positive and large errors are penalized
more heavily.
</p>

<hr>

<h2>7. Gradient Descent</h2>
<p>
Gradient Descent is an optimization algorithm used to minimize the cost function.
It works iteratively to find the best values for model parameters.
</p>

<h3>Key Idea</h3>
<p>
The gradient of a function points in the direction of maximum increase.
To minimize the function, we move in the opposite direction of the gradient.
</p>

<h3>Steps in Gradient Descent</h3>
<ol>
  <li>Choose a random starting point for parameters</li>
  <li>Compute the gradient of the cost function</li>
  <li>Move parameters in the opposite direction of the gradient</li>
  <li>Repeat until the cost function stops decreasing</li>
</ol>

<hr>

<h2>8. Learning Rate (α)</h2>
<p>
The learning rate controls how much the parameters change in each step of
gradient descent.
</p>

<ul>
  <li>Too large → may overshoot the minimum</li>
  <li>Too small → very slow convergence</li>
  <li>Optimal value → fast and stable learning</li>
</ul>

<hr>

<h2>9. Gradient Descent Mathematics (Slope Optimization)</h2>
<p>
For a simple model where only the slope is optimized:
</p>

<pre>
Cost Function = (y[i] - m * x[i])²
</pre>

<p>
Taking derivative with respect to m:
</p>

<pre>
d/dm (y[i] - m * x[i])²
= -2 (y[i] - m * x[i]) * x[i]
</pre>

<p>
This derivative is used to update the slope value.
</p>

<hr>

<h2>10. Direct Formula for Slope and Intercept</h2>

<h3>Slope (b₁)</h3>
<pre>
b₁ = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
</pre>

<h3>Intercept (b₀)</h3>
<pre>
b₀ = ȳ - b₁ * x̄
</pre>

<p>
Here, x̄ and ȳ represent the mean values of X and Y respectively.
</p>

<hr>

<h2>11. RMSE (Root Mean Square Error)</h2>
<p>
RMSE measures the average magnitude of prediction errors.
</p>

<pre>
RMSE = √(1/m Σ(y - ŷ)²)
</pre>

<p>
Lower RMSE indicates a better model.
</p>

<hr>

<h2>12. R² Score (Coefficient of Determination)</h2>
<p>
R² score measures how well the regression line fits the data.
</p>

<pre>
R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
</pre>

<h3>Interpretation</h3>
<ul>
  <li>R² ≈ 1 → Very good model</li>
  <li>R² far from 1 → Poor model</li>
</ul>

<hr>

<h2>13. Multiple Linear Regression</h2>
<p>
Multiple Linear Regression is an extension of simple linear regression where
there are multiple independent variables and one dependent variable.
</p>

<h3>Equation</h3>
<pre>
Y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
</pre>

<p>
By introducing x₀ = 1, the equation becomes:
</p>

<pre>
Y = β₀x₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
</pre>

<hr>

<h2>14. Matrix Representation</h2>
<p>
The multiple linear regression equation can be written in matrix form:
</p>

<pre>
Y = βᵀX
</pre>

<p>
This representation is efficient for handling large datasets.
</p>

<hr>

<h2>15. Cost Function in Multiple Linear Regression</h2>
<p>
The hypothesis function is:
</p>

<pre>
h<sub>β</sub>(x) = βᵀx
</pre>

<p>
The cost function is defined as:
</p>

<pre>
J(β) = (1 / 2m) Σ(h<sub>β</sub>(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
</pre>

<hr>

<h2>16. Gradient Descent for Multiple Linear Regression</h2>

<h3>Initialization</h3>
<p>
All β values are initialized to zero.
</p>

<h3>Update Rule</h3>
<pre>
β<sub>j</sub> := β<sub>j</sub> - α ∂J(β)/∂β<sub>j</sub>
</pre>

<p>
After applying mathematics, the update equation becomes:
</p>

<pre>
β<sub>j</sub> := β<sub>j</sub> - α (1/m) Σ(h<sub>β</sub>(x⁽ⁱ⁾) - y⁽ⁱ⁾) x<sub>j</sub>⁽ⁱ⁾
</pre>

<hr>

<h2>17. Batch Gradient Descent</h2>
<p>
This method of gradient descent uses the entire dataset to update parameters
in each iteration. It is called <b>Batch Gradient Descent</b>.
</p>

<p>
It is stable and guarantees convergence for convex cost functions, but can be
computationally expensive for large datasets.
</p>

<hr>

<h2>Simple Linear Regression</h2>

<p>
<strong>Simple Linear Regression (SLR)</strong> is a statistical and machine learning technique
used to model the <strong>linear relationship</strong> between two variables:
</p>

<ul>
  <li><strong>One independent variable</strong> — often called feature (X)</li>
  <li><strong>One dependent variable</strong> — target (Y)</li>
</ul>

<p>
It assumes that changes in <strong>X</strong> result in proportional changes in <strong>Y</strong>,
and fits a straight line through the observed data to make predictions.
</p>

<hr>

<h2>🧠 Model Equation</h2>

<p>The model is represented by the equation:</p>

<p style="font-size:18px;">
  <strong>y = β<sub>0</sub> + β<sub>1</sub>x + ε</strong>
</p>

<h4>Where:</h4>
<ul>
  <li><strong>y</strong> = dependent variable (value we want to predict)</li>
  <li><strong>x</strong> = independent variable (input)</li>
  <li><strong>β<sub>0</sub></strong> = intercept (predicted value of y when x = 0)</li>
  <li><strong>β<sub>1</sub></strong> = slope/coefficient (how strongly x affects y)</li>
  <li><strong>ε</strong> = error term (difference between actual and predicted values)</li>
</ul>

<p>
👉 The goal is to choose values of <strong>β<sub>0</sub></strong> and <strong>β<sub>1</sub></strong>
so that the predicted line fits the data as closely as possible.
</p>



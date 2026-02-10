
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

<h1>Multiple Linear Regression (MLR) & Deployment on Render</h1>

<hr>

<h2>PART 1️⃣ What is Multiple Linear Regression (MLR)?</h2>

<h3>🔹 Definition (Simple words)</h3>
<p>
<b>Multiple Linear Regression</b> is a supervised machine learning algorithm used to
<b>predict a numeric value</b> using <b>more than one input feature</b>.
</p>

<p>👉 It is an extension of <b>Simple Linear Regression</b>.</p>

<hr>

<h3>🔹 Mathematical Equation</h3>

<p><b>Simple Linear Regression:</b></p>
<p><code>y = mx + c</code></p>

<p><b>Multiple Linear Regression:</b></p>
<p>
<code>
y = b<sub>0</sub> + b<sub>1</sub>x<sub>1</sub> + b<sub>2</sub>x<sub>2</sub> + 
b<sub>3</sub>x<sub>3</sub> + ... + b<sub>n</sub>x<sub>n</sub>
</code>
</p>

<p><b>Where:</b></p>
<ul>
  <li><b>y</b> → Predicted output (target)</li>
  <li><b>x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>...</b> → Input features</li>
  <li><b>b<sub>0</sub></b> → Intercept</li>
  <li><b>b<sub>1</sub>, b<sub>2</sub>, b<sub>3</sub>...</b> → Coefficients (weights)</li>
</ul>

<hr>

<h3>🔹 Example (Real-world)</h3>

<p><b>Predict House Price using:</b></p>
<ul>
  <li>Area</li>
  <li>Number of bedrooms</li>
  <li>Location</li>
  <li>Age of house</li>
</ul>

<p><b>Mapping:</b></p>
<ul>
  <li>Area → x<sub>1</sub></li>
  <li>Bedrooms → x<sub>2</sub></li>
  <li>Location → x<sub>3</sub></li>
  <li>Age → x<sub>4</sub></li>
</ul>

<p>
The model learns how <b>each feature contributes</b> to the house price.
</p>

<hr>

<h3>🔹 Why we use MLR?</h3>
<ul>
  <li>✅ Uses multiple factors</li>
  <li>✅ More accurate than Simple Linear Regression</li>
  <li>✅ Widely used in salary, price, and demand prediction</li>
</ul>

<hr>

<h3>🔹 Training an MLR Model (Concept)</h3>
<ol>
  <li>Load dataset</li>
  <li>Split data into <b>X (features)</b> and <b>y (target)</b></li>
  <li>Apply <code>train_test_split</code></li>
  <li>Train model using <code>LinearRegression()</code></li>
  <li>Evaluate using:
    <ul>
      <li>R² Score</li>
      <li>Mean Squared Error (MSE)</li>
    </ul>
  </li>
</ol>

<hr



<h1>📈 Polynomial Linear Regression</h1>

<h2>🔍 What is Polynomial Linear Regression?</h2>
<p>
Polynomial Linear Regression is a supervised machine learning regression technique
used when the relationship between the independent variable (X) and dependent
variable (Y) is <b>non-linear</b>, but the model is still <b>linear in parameters</b>.
</p>

<p><b>Key Idea:</b> The model is linear with respect to coefficients, but non-linear with respect to input features.</p>

<hr>

<h2>❓ Why Do We Need Polynomial Regression?</h2>
<p>
Simple Linear Regression assumes a straight-line relationship:
</p>

<p><b>y = mx + c</b></p>

<p>
However, real-world data often follows curved patterns such as:
</p>

<ul>
  <li>U-shaped curves</li>
  <li>Parabolic trends</li>
  <li>Complex growth patterns</li>
</ul>

<p>
In such cases, linear regression underfits the data.
Polynomial regression solves this problem by adding higher-degree terms.
</p>

<hr>

<h2>🧮 Mathematical Representation</h2>

<p><b>Degree 2 (Quadratic):</b></p>
<p>y = b₀ + b₁x + b₂x²</p>

<p><b>Degree 3 (Cubic):</b></p>
<p>y = b₀ + b₁x + b₂x² + b₃x³</p>

<p><b>General form:</b></p>
<p>y = b₀ + b₁x + b₂x² + ... + bₙxⁿ</p>

<p>
Even though higher powers of x exist, the equation is still linear
because coefficients are not multiplied together.
</p>

<hr>

<h2>🧠 Why Is It Called Linear Regression?</h2>
<p>
It is called linear regression because the model is linear with respect to the
parameters (b₀, b₁, b₂, ...), not the input features.
</p>

<hr>

<h2>⚙️ How Polynomial Regression Works</h2>

<ol>
  <li>Start with the original feature X</li>
  <li>Create polynomial features (x², x³, ...)</li>
  <li>Apply linear regression on the transformed features</li>
</ol>

<hr>

<h2>🧑‍💻 Python Implementation (scikit-learn)</h2>

<pre>
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)
</pre>

<hr>

<h2>🎯 Choosing the Polynomial Degree</h2>

<table border="1" cellpadding="6">
  <tr>
    <th>Degree</th>
    <th>Effect</th>
  </tr>
  <tr>
    <td>Low</td>
    <td>Underfitting</td>
  </tr>
  <tr>
    <td>Optimal</td>
    <td>Good bias-variance balance</td>
  </tr>
  <tr>
    <td>High</td>
    <td>Overfitting</td>
  </tr>
</table>

<hr>

<h2>⚠️ Overfitting Problem</h2>
<p>
High-degree polynomials may perfectly fit training data but perform poorly
on unseen data.
</p>

<p><b>Solutions:</b></p>
<ul>
  <li>Cross-validation</li>
  <li>Regularization (Ridge, Lasso)</li>
  <li>Choosing optimal degree</li>
</ul>

<hr>

<h2>✅ Advantages</h2>
<ul>
  <li>Models non-linear relationships</li>
  <li>Easy to implement</li>
  <li>Flexible and powerful</li>
</ul>

<h2>❌ Disadvantages</h2>
<ul>
  <li>Prone to overfitting</li>
  <li>Sensitive to outliers</li>
  <li>Poor extrapolation</li>
</ul>

<hr>

<h2>🆚 Linear vs Polynomial Regression</h2>

<table border="1" cellpadding="6">
  <tr>
    <th>Feature</th>
    <th>Linear</th>
    <th>Polynomial</th>
  </tr>
  <tr>
    <td>Relationship</td>
    <td>Straight line</td>
    <td>Curved</td>
  </tr>
  <tr>
    <td>Complexity</td>
    <td>Low</td>
    <td>Higher</td>
  </tr>
  <tr>
    <td>Accuracy on non-linear data</td>
    <td>Low</td>
    <td>High</td>
  </tr>
</table>

<hr>

<h2>📌 Real-World Applications</h2>
<ul>
  <li>Salary prediction</li>
  <li>House price prediction</li>
  <li>Growth trend analysis</li>
  <li>Engineering and physics models</li>
</ul>

<hr>

<h2>📝 Summary</h2>
<p>
Polynomial Linear Regression extends linear regression by introducing
polynomial features to capture non-linear patterns while keeping the model linear.
Choosing the correct degree is critical for good performance.
</p>


<h1>📊 R² Score and Adjusted R² Score</h1>

<h2>🔍 What is R² Score?</h2>
<p>
R² Score (Coefficient of Determination) measures how well a regression model
explains the variation in the dependent variable (Y).
</p>

<p><b>In simple words:</b></p>
<p>
It tells us what percentage of the total variation in the target variable
is explained by the model.
</p>

<hr>

<h2>🧠 Intuition Behind R² Score</h2>
<p>
If your model predicts values close to the actual data points,
the R² score will be high.
</p>

<p>
For example, an R² score of <b>0.85</b> means:
</p>
<ul>
  <li>85% of the variation in Y is explained by the model</li>
  <li>15% remains unexplained</li>
</ul>

<hr>

<h2>📐 Formula of R² Score</h2>

<p>
<b>R² = 1 − (SS<sub>res</sub> / SS<sub>tot</sub>)</b>
</p>

<p><b>Where:</b></p>
<ul>
  <li><b>SS<sub>tot</sub></b> – Total Sum of Squares (total variation in Y)</li>
  <li><b>SS<sub>res</sub></b> – Residual Sum of Squares (prediction error)</li>
</ul>

<hr>

<h2>📊 R² Score Interpretation</h2>

<table border="1" cellpadding="6">
  <tr>
    <th>R² Value</th>
    <th>Interpretation</th>
  </tr>
  <tr>
    <td>1.0</td>
    <td>Perfect model</td>
  </tr>
  <tr>
    <td>0.9</td>
    <td>Very good fit</td>
  </tr>
  <tr>
    <td>0.7</td>
    <td>Good fit</td>
  </tr>
  <tr>
    <td>0.0</td>
    <td>No explanatory power</td>
  </tr>
  <tr>
    <td>&lt; 0</td>
    <td>Worse than predicting mean</td>
  </tr>
</table>

<hr>

<h2>⚠️ Limitation of R² Score</h2>
<p>
R² score always increases when new features are added,
even if those features are useless.
</p>

<p>
This makes R² unreliable for Multiple Linear Regression.
</p>

<hr>

<h2>📉 What is Adjusted R² Score?</h2>
<p>
Adjusted R² improves R² by penalizing unnecessary independent variables.
</p>

<p>
It increases only when a new feature actually improves the model.
</p>

<hr>

<h2>📐 Formula of Adjusted R²</h2>

<p>
<b>
Adjusted R² = 1 − [(1 − R²) × (n − 1) / (n − p − 1)]
</b>
</p>

<p><b>Where:</b></p>
<ul>
  <li><b>n</b> = number of observations</li>
  <li><b>p</b> = number of independent variables</li>
  <li><b>R²</b> = R² score</li>
</ul>

<hr>

<h2>🧠 Why Adjusted R² is Important</h2>
<ul>
  <li>Penalizes irrelevant features</li>
  <li>Prevents overfitting</li>
  <li>More reliable for multiple regression</li>
</ul>

<hr>

<h2>🆚 R² Score vs Adjusted R² Score</h2>

<table border="1" cellpadding="6">
  <tr>
    <th>Feature</th>
    <th>R² Score</th>
    <th>Adjusted R² Score</th>
  </tr>
  <tr>
    <td>Penalizes extra features</td>
    <td>No</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Always increases</td>
    <td>Yes</td>
    <td>No</td>
  </tr>
  <tr>
    <td>Best for</td>
    <td>Simple Linear Regression</td>
    <td>Multiple Linear Regression</td>
  </tr>
  <tr>
    <td>Reliability</td>
    <td>Medium</td>
    <td>High</td>
  </tr>
</table>

<hr>

<h2>🧪 Python Example</h2>

<pre>
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R2:", adjusted_r2)
</pre>

<hr>

<h2>📌 When to Use Which?</h2>

<ul>
  <li>Use <b>R²</b> for Simple Linear Regression</li>
  <li>Use <b>Adjusted R²</b> for Multiple Linear Regression</li>
  <li>Use Adjusted R² for model comparison</li>
</ul>

<hr>

<h2>📝 Summary</h2>
<p>
R² score measures how well a regression model fits the data,
while Adjusted R² provides a more reliable measure by considering
the number of features used in the model.
</p>

<h1>🎯 Regularization in Machine Learning</h1>

<h2>🔍 What is Regularization?</h2>
<p>
Regularization is a technique used to prevent <b>overfitting</b> in machine learning models
by adding a <b>penalty term</b> to the loss function.
</p>

<p>
In simple words, regularization discourages the model from learning
complex patterns by forcing the coefficients to remain small.
</p>

<hr>

<h2>❓ Why Do We Need Regularization?</h2>
<p>
Overfitting occurs when a model:
</p>
<ul>
  <li>Performs very well on training data</li>
  <li>Performs poorly on unseen test data</li>
  <li>Learns noise instead of the real pattern</li>
</ul>

<p>
This problem is common in:
</p>
<ul>
  <li>Polynomial Regression</li>
  <li>Multiple Linear Regression</li>
  <li>High-dimensional datasets</li>
</ul>

<hr>

<h2>🧠 Intuition Behind Regularization</h2>
<p>
Large coefficient values make the model overly sensitive to small changes in input data.
Regularization penalizes large coefficients and keeps the model simple and stable.
</p>

<hr>

<h2>🧮 Cost Function Without Regularization</h2>
<p>
<b>Loss = Σ (y − ŷ)²</b>
</p>

<p>
This minimizes prediction error but does not control model complexity.
</p>

<hr>

<h2>➕ Cost Function With Regularization</h2>
<p>
<b>Loss = Σ (y − ŷ)² + Penalty Term</b>
</p>

<p>
The penalty term depends on the type of regularization used.
</p>

<hr>

<h2>🔥 Types of Regularization</h2>
<ul>
  <li>Ridge Regression (L2)</li>
  <li>Lasso Regression (L1)</li>
 
</ul>

<hr>

<h2>1️⃣ Ridge Regression (L2 Regularization)</h2>
<p>
Ridge regression adds the square of the coefficients as a penalty term.
</p>

<p><b>Formula:</b></p>
<p>
Loss = Σ (y − ŷ)² + λ Σ w²
</p>

<ul>
  <li>Shrinks coefficients toward zero</li>
  <li>Does not eliminate features</li>
  <li>Handles multicollinearity well</li>
</ul>

<hr>

<h2>2️⃣ Lasso Regression (L1 Regularization)</h2>
<p>
Lasso regression adds the absolute value of the coefficients as a penalty.
</p>

<p><b>Formula:</b></p>
<p>
Loss = Σ (y − ŷ)² + λ Σ |w|
</p>

<ul>
  <li>Shrinks coefficients</li>
  <li>Can make coefficients exactly zero</li>
  <li>Performs feature selection</li>
</ul>



<h2>🎛️ Role of Lambda (λ)</h2>
<p>
Lambda controls the strength of regularization.
</p>

<table border="1" cellpadding="6">
  <tr>
    <th>λ Value</th>
    <th>Effect</th>
  </tr>
  <tr>
    <td>0</td>
    <td>No regularization</td>
  </tr>
  <tr>
    <td>Small</td>
    <td>Slight penalty</td>
  </tr>
  <tr>
    <td>Large</td>
    <td>Strong penalty</td>
  </tr>
  <tr>
    <td>Very large</td>
    <td>Underfitting</td>
  </tr>
</table>

<hr>

<h2>⚖️ Bias–Variance Tradeoff</h2>
<p>
Regularization slightly increases bias but significantly reduces variance,
resulting in better generalization.
</p>

<hr>

<h2>🧪 Python Example (scikit-learn)</h2>

<pre>
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)


</pre>

<hr>

<h2>🆚 Regularization Comparison</h2>

<table border="1" cellpadding="6">
  <tr>
    <th>Feature</th>
    <th>Ridge</th>
    <th>Lasso</th>
    <th>Elastic Net</th>
  </tr>
  <tr>
    <td>Penalty Type</td>
    <td>L2</td>
    <td>L1</td>
    <td>L1 + L2</td>
  </tr>
  <tr>
    <td>Feature Selection</td>
    <td>No</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>Handles Multicollinearity</td>
    <td>Yes</td>
    <td>No</td>
    <td>Yes</td>
  </tr>
</table>

<hr>

<h2>📌 When to Use Regularization?</h2>
<ul>
  <li>When the model overfits</li>
  <li>When dataset has many features</li>
  <li>When using polynomial features</li>
  <li>When multicollinearity exists</li>
</ul>

<hr>

<h2>📝 Summary</h2>
<p>
Regularization helps build simpler, more generalizable models by penalizing
large coefficients. Ridge, Lasso are the most commonly used
regularization techniques in regression problems.
</p>

<h1>📌 Classification and K-Nearest Neighbors (KNN) Algorithm</h1>

<hr>

<h2>1️⃣ What is Classification?</h2>
<p>
Classification is a <b>supervised machine learning technique</b> where the target variable is
<b>categorical</b>.
</p>

<p><b>Examples:</b></p>
<ul>
  <li>Spam / Not Spam</li>
  <li>Yes / No</li>
  <li>Malignant / Benign</li>
</ul>

<p>
In this project, the dataset is a <b>binary classification problem</b> where:
</p>
<ul>
  <li><b>1 → Malignant</b></li>
  <li><b>0 → Benign</b></li>
</ul>

<hr>

<h2>2️⃣ What is K-Nearest Neighbors (KNN)?</h2>
<p>
KNN is a <b>distance-based supervised learning algorithm</b> used for classification.
</p>

<ul>
  <li>It does not learn a mathematical equation</li>
  <li>It stores the training data</li>
  <li>Prediction is made using nearest data points</li>
</ul>

<p>
KNN is also called a <b>lazy learning algorithm</b>.
</p>

<hr>

<h2>3️⃣ How KNN Works</h2>

<ol>
  <li>Select the value of <b>K</b> (number of neighbors)</li>
  <li>Calculate distance between test point and all training points</li>
  <li>Select <b>K nearest neighbors</b></li>
  <li>Apply <b>majority voting</b></li>
</ol>

<p>
The class with the highest votes becomes the predicted output.
</p>

<hr>

<h2>4️⃣ Distance Formula (Euclidean Distance)</h2>

<p>
KNN uses <b>Euclidean Distance</b> by default.
</p>

<p><b>Formula:</b></p>

<p>
d = √[(x₁ − x₂)² + (y₁ − y₂)² + ...]
</p>

<p>
For multiple features:
</p>

<p>
d = √(Σ(xᵢ − yᵢ)²)
</p>

<p>
In this dataset, distance is calculated using <b>30 numerical features</b>.
</p>

<hr>

<h2>5️⃣ Default Value of K</h2>

<p>
In Scikit-learn:
</p>

<pre>
KNeighborsClassifier()
</pre>

<p>
The default value of <b>K = 5</b>.
</p>

<p>
Using K=5 helps reduce overfitting compared to K=1.
</p>

<hr>

<h2>6️⃣ Confusion Matrix</h2>

<p>
A confusion matrix evaluates the performance of a classification model.
</p>

<table border="1" cellpadding="6">
  <tr>
    <th></th>
    <th>Predicted 0</th>
    <th>Predicted 1</th>
  </tr>
  <tr>
    <th>Actual 0</th>
    <td>True Negative (TN)</td>
    <td>False Positive (FP)</td>
  </tr>
  <tr>
    <th>Actual 1</th>
    <td>False Negative (FN)</td>
    <td>True Positive (TP)</td>
  </tr>
</table>

<p>
In medical datasets, <b>false negatives</b> are very dangerous because cancer cases may be missed.
</p>

<hr>

<h2>7️⃣ Accuracy</h2>

<p>
Accuracy measures the overall correctness of the model.
</p>

<p><b>Formula:</b></p>

<p>
Accuracy = (TP + TN) / (TP + TN + FP + FN)
</p>

<p>
Accuracy alone is not reliable for imbalanced datasets.
</p>

<hr>

<h2>8️⃣ Classification Report</h2>

<p>
The classification report provides:
</p>

<ul>
  <li>Precision</li>
  <li>Recall</li>
  <li>F1-Score</li>
  <li>Support</li>
</ul>

<h3>🔹 Precision</h3>
<p>
Precision = TP / (TP + FP)
</p>
<p>
It measures how many predicted positive cases are actually positive.
</p>

<h3>🔹 Recall (Sensitivity)</h3>
<p>
Recall = TP / (TP + FN)
</p>
<p>
It measures how many actual positive cases are correctly identified.
</p>

<h3>🔹 F1 Score</h3>
<p>
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
</p>
<p>
It balances precision and recall.
</p>

<h3>🔹 Support</h3>
<p>
Support represents the number of actual samples in each class.
</p>

<hr>

<h2>9️⃣ Selecting the Best K Value</h2>

<p>
To find the optimal K:
</p>

<ol>
  <li>Train the model for different K values</li>
  <li>Calculate accuracy or error for each K</li>
  <li>Select K with highest test accuracy or lowest error</li>
</ol>

<p>
Odd values of K are preferred to avoid ties in binary classification.
</p>

<ul>
  <li>Small K → Overfitting</li>
  <li>Large K → Underfitting</li>
  <li>Optimal K → Best balance</li>
</ul>

<hr>

<h2>🔚 Conclusion</h2>

<ul>
  <li>Classification predicts categorical outputs</li>
  <li>KNN is a distance-based, lazy learning algorithm</li>
  <li>Uses Euclidean distance</li>
  <li>Evaluation includes confusion matrix and classification report</li>
  <li>Best K is selected by testing multiple values</li>
</ul>


<h1>Naive Bayes Classification – Detailed Explanation</h1>

<h2>1. Introduction to Naive Bayes</h2>
<p>
Naive Bayes is a <b>probabilistic classification algorithm</b> based on 
<b>Bayes’ Theorem</b>. It is called <i>naive</i> because it assumes that all 
features are <b>independent of each other</b> given the class label.
</p>

<p>
Despite this strong assumption, Naive Bayes performs very well in many real-world
applications such as spam detection, sentiment analysis, and text classification.
</p>

<hr>

<h2>2. Bayes’ Theorem</h2>
<p>The core formula behind Naive Bayes is:</p>

<p><b>P(C | X) = (P(X | C) × P(C)) / P(X)</b></p>

<ul>
  <li><b>C</b> – Class label</li>
  <li><b>X</b> – Input features (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>)</li>
  <li><b>P(C | X)</b> – Posterior probability</li>
  <li><b>P(C)</b> – Prior probability</li>
  <li><b>P(X | C)</b> – Likelihood</li>
  <li><b>P(X)</b> – Evidence (constant, usually ignored)</li>
</ul>

<p>
Using the naive assumption, the likelihood becomes:
</p>

<p>
<b>P(X | C) = P(x<sub>1</sub> | C) × P(x<sub>2</sub> | C) × ... × P(x<sub>n</sub> | C)</b>
</p>

<hr>

<h2>3. How Naive Bayes Divides the Data</h2>

<h3>3.1 Dataset Structure</h3>
<p>
The dataset is divided into:
</p>
<ul>
  <li><b>Independent variables (X)</b> – features</li>
  <li><b>Dependent variable (y)</b> – class label(s)</li>
</ul>

<h3>3.2 Train-Test Split</h3>
<p>
The dataset is split into training and testing sets:
</p>
<ul>
  <li><b>Training data</b> – used to learn probabilities</li>
  <li><b>Testing data</b> – used to evaluate the model</li>
</ul>

<p>
Typically, 70–80% of data is used for training and 20–30% for testing.
</p>

<hr>

<h2>4. How Naive Bayes is Trained</h2>

<p>
Naive Bayes does not use gradient descent or weight optimization. Instead, it
learns by <b>calculating probabilities</b>.
</p>

<h3>4.1 Prior Probability</h3>
<p>
The prior probability of a class is calculated as:
</p>

<p>
<b>P(C) = (Number of samples in class C) / (Total number of samples)</b>
</p>

<h3>4.2 Likelihood Calculation</h3>
<p>
The likelihood depends on the type of Naive Bayes used:
</p>

<ul>
  <li><b>Gaussian Naive Bayes</b> – continuous data</li>
  <li><b>Multinomial Naive Bayes</b> – count-based data (text)</li>
  <li><b>Bernoulli Naive Bayes</b> – binary data (0/1)</li>
</ul>

<h3>Gaussian Naive Bayes</h3>
<p>
Gaussian Naive Bayes assumes features follow a normal distribution:
</p>

<p>
<b>
P(x | C) = (1 / √(2πσ²)) × exp(−(x − μ)² / (2σ²))
</b>
</p>

<p>
For each feature and each class, the algorithm calculates:
</p>
<ul>
  <li>Mean (μ)</li>
  <li>Variance (σ²)</li>
</ul>

<hr>

<h2>5. Normalization in Naive Bayes</h2>

<p>
Normalization is <b>not mandatory</b> but recommended when features have different
scales.
</p>

<p>
Since Gaussian Naive Bayes uses mean and variance, large feature values can
dominate probability calculations.
</p>

<p>
Standardization (mean = 0, variance = 1) improves numerical stability and model
performance.
</p>

<hr>

<h2>6. How Naive Bayes Makes Predictions</h2>

<p>
For a new input sample, Naive Bayes performs the following steps:
</p>

<ol>
  <li>Calculate posterior probability for each class</li>
  <li>Compare probabilities</li>
  <li>Select the class with the highest probability</li>
</ol>

<p>
The predicted class is:
</p>

<p>
<b>ŷ = argmax<sub>C</sub> P(C | X)</b>
</p>

<hr>

<h2>7. Multi-Label Classification with Naive Bayes</h2>

<h3>7.1 What is Multi-Label Classification?</h3>
<p>
In multi-label classification, a single data point can belong to
<b>multiple classes at the same time</b>.
</p>

<p>
Example:
</p>
<ul>
  <li>Email → Spam, Promotion</li>
  <li>Image → Dog, Outdoor, Daytime</li>
</ul>

<h3>7.2 One-vs-Rest Strategy</h3>
<p>
Naive Bayes is inherently a single-label classifier. For multi-label problems,
the <b>One-vs-Rest (OvR)</b> strategy is used.
</p>

<ul>
  <li>One classifier is trained per label</li>
  <li>Each classifier predicts presence (1) or absence (0) of that label</li>
</ul>

<p>
The final output is a binary vector representing all predicted labels.
</p>

<hr>

<h2>8. Model Evaluation and Classification Report</h2>

<p>
After prediction, model performance is evaluated using a
<b>classification report</b>.
</p>

<h3>Evaluation Metrics</h3>

<ul>
  <li>
    <b>Precision</b> = TP / (TP + FP)  
    <br>Measures correctness of predicted labels
  </li>
  <li>
    <b>Recall</b> = TP / (TP + FN)  
    <br>Measures how many actual labels were detected
  </li>
  <li>
    <b>F1-Score</b>  
    <br>Harmonic mean of precision and recall
  </li>
  <li>
    <b>Support</b>  
    <br>Number of true samples for each label
  </li>
</ul>

<h3>Averaging Methods</h3>
<ul>
  <li><b>Micro Average</b> – global performance</li>
  <li><b>Macro Average</b> – average per label</li>
  <li><b>Weighted Average</b> – accounts for label imbalance</li>
</ul>

<hr>

<h2>9. Overall Workflow</h2>

<pre>
Dataset
   ↓
Train-Test Split
   ↓
(Optional) Normalization
   ↓
Probability Calculation (Training)
   ↓
Prediction using Bayes Theorem
   ↓
Evaluation using Classification Report
</pre>

<hr>

<h2>10. Advantages of Naive Bayes</h2>

<ul>
  <li>Fast training and prediction</li>
  <li>Works well with small datasets</li>
  <li>Handles high-dimensional data efficiently</li>
  <li>Simple and interpretable</li>
</ul>

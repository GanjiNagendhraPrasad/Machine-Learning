
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


<h1>Logistic Regression – Detailed Explanation</h1>

<h2>1. What is Logistic Regression?</h2>
<p>
Logistic Regression is a <b>supervised machine learning algorithm</b> used for
<b>classification problems</b>. Despite its name, it is not used for regression.
Instead, it predicts the <b>probability</b> that a given input belongs to a
particular class.
</p>

<p>
It is mainly used for:
</p>
<ul>
  <li>Binary Classification (0 or 1, Yes or No)</li>
  <li>Multi-class Classification (using One-vs-Rest strategy)</li>
</ul>

---

<h2>2. Why Not Linear Regression for Classification?</h2>
<p>
Linear Regression produces outputs that range from <b>-∞ to +∞</b>, which are not
suitable for classification. Classification problems require outputs between
<b>0 and 1</b> so they can be interpreted as probabilities.
</p>

<p>
Logistic Regression solves this by using the <b>Sigmoid (Logistic) Function</b>.
</p>

---

<h2>3. Sigmoid Function</h2>
<p>
The sigmoid function converts any real value into a number between 0 and 1.
</p>

<pre>
σ(z) = 1 / (1 + e<sup>-z</sup>)
</pre>

<p>
Where:
</p>
<pre>
z = m1x1 + m2x2 + ... + mnxn + c
</pre>

<p>
This output is treated as a probability:
</p>
<ul>
  <li>Probability ≥ 0.5 → Class 1</li>
  <li>Probability &lt; 0.5 → Class 0</li>
</ul>

---

<h2>4. Cost Function (Log Loss)</h2>
<p>
Logistic Regression uses <b>Log Loss (Binary Cross-Entropy)</b> instead of Mean
Squared Error.
</p>

<pre>
Loss = -[y log(p) + (1 - y) log(1 - p)]
</pre>

<p>
This cost function penalizes wrong predictions heavily when the model is very
confident but incorrect.
</p>

---

<h2>5. Training Process</h2>
<ol>
  <li>Initialize weights</li>
  <li>Calculate predicted probabilities using sigmoid</li>
  <li>Compute loss using log loss</li>
  <li>Update weights using optimization</li>
  <li>Repeat until loss is minimized</li>
</ol>

---

<h2>6. Multi-Class Logistic Regression</h2>
<p>
For multi-class problems, Logistic Regression uses the
<b>One-vs-Rest (OvR)</b> strategy. A separate classifier is trained for each class,
and the class with the highest probability is selected.
</p>

---

<hr>

<h1>Project Explanation: Iris Flower Classification</h1>

<h2>1. Dataset</h2>
<p>
The project uses the famous <b>Iris dataset</b>, which contains measurements of
iris flowers belonging to three species:
</p>

<ul>
  <li>Iris-setosa</li>
  <li>Iris-versicolor</li>
  <li>Iris-virginica</li>
</ul>

<p>
The dataset includes the following features:
</p>
<ul>
  <li>Sepal Length</li>
  <li>Sepal Width</li>
  <li>Petal Length</li>
  <li>Petal Width</li>
</ul>

---

<h2>2. Data Preprocessing</h2>
<p>
The <b>Id</b> column is removed since it does not contribute to prediction.
</p>

<p>
The target column (<b>Species</b>) is label-encoded:
</p>
<ul>
  <li>Iris-setosa → 0</li>
  <li>Iris-versicolor → 1</li>
  <li>Iris-virginica → 2</li>
</ul>

---

<h2>3. Feature and Target Separation</h2>
<p>
The dataset is divided into:
</p>
<ul>
  <li><b>X (Features)</b>: Sepal and petal measurements</li>
  <li><b>y (Target)</b>: Encoded flower species</li>
</ul>

---

<h2>4. Train-Test Split</h2>
<p>
The data is split into training and testing sets:
</p>
<ul>
  <li>80% Training Data</li>
  <li>20% Testing Data</li>
</ul>

<p>
This ensures that the model is evaluated on unseen data.
</p>

---

<h2>5. Model Training</h2>
<p>
A Logistic Regression model from <b>scikit-learn</b> is used and trained on the
training dataset.
</p>

<p>
Internally, the model:
</p>
<ul>
  <li>Applies sigmoid and softmax functions</li>
  <li>Uses One-vs-Rest classification</li>
  <li>Learns optimal feature weights</li>
</ul>

---

<h2>6. Model Evaluation</h2>
<p>
Model performance is evaluated using <b>accuracy score</b> on both:
</p>
<ul>
  <li>Training Data</li>
  <li>Testing Data</li>
</ul>

<p>
High accuracy on test data indicates good generalization.
</p>

---

<h2>7. Prediction with Custom Input</h2>
<p>
The model allows prediction using user-defined flower measurements.
</p>

<p>
Based on the predicted class value:
</p>
<ul>
  <li>0 → Setosa</li>
  <li>1 → Versicolor</li>
  <li>2 → Virginica</li>
</ul>

---

<h2>8. Conclusion</h2>
<p>
This project demonstrates the effective use of Logistic Regression for
multi-class classification. It includes proper data preprocessing, model
training, evaluation, and real-time prediction, making it suitable for both
academic and practical applications.
</p>

<h1>Decision Tree – Detailed Explanation</h1>

<h2>1. What is a Decision Tree?</h2>
<p>
A <b>Decision Tree</b> is a <b>supervised machine learning algorithm</b> used for
both <b>classification</b> and <b>regression</b> problems. It works by splitting
the dataset into smaller subsets based on feature values, forming a tree-like
structure of decisions.
</p>

<p>
Each internal node represents a <b>feature</b>, each branch represents a
<b>decision rule</b>, and each leaf node represents the <b>final output</b>.
</p>

---

<h2>2. Why Decision Trees?</h2>
<ul>
  <li>Easy to understand and interpret</li>
  <li>Works with both numerical and categorical data</li>
  <li>No need for feature scaling</li>
  <li>Handles non-linear relationships well</li>
</ul>

---

<h2>3. How a Decision Tree Works</h2>
<p>
A decision tree repeatedly splits the data based on the feature that provides
the <b>best separation</b> between classes.
</p>

<p>
For classification problems, common splitting criteria are:
</p>
<ul>
  <li><b>Gini Index</b></li>
  <li><b>Entropy</b></li>
  <li><b>Information Gain</b></li>
</ul>

---

<h2>4. Entropy</h2>
<p>
Entropy measures the <b>impurity</b> or randomness in the data.
</p>

<pre>
Entropy = - Σ p log2(p)
</pre>

<p>
Where <b>p</b> is the probability of each class.
Lower entropy means purer data.
</p>

---

<h2>5. Information Gain</h2>
<p>
Information Gain measures how much entropy is reduced after a split.
The feature with the <b>highest information gain</b> is selected for splitting.
</p>

<pre>
Information Gain = Entropy(parent) − Weighted Entropy(children)
</pre>

---

<h2>6. Training Process</h2>
<ol>
  <li>Select the best feature using entropy or gini</li>
  <li>Split the dataset based on that feature</li>
  <li>Repeat recursively for each child node</li>
  <li>Stop when data becomes pure or stopping criteria is met</li>
</ol>

---

<h2>7. Overfitting in Decision Trees</h2>
<p>
Decision trees can easily overfit the training data.
To prevent this, we use:
</p>
<ul>
  <li>Maximum depth</li>
  <li>Minimum samples per split</li>
  <li>Minimum samples per leaf</li>
  <li>Pruning</li>
</ul>

---

<hr>

<h1>Project Explanation: Decision Tree Classification</h1>

<h2>1. Dataset</h2>
<p>
This project uses the <b>Iris dataset</b>, which contains flower measurements
belonging to three different species:
</p>

<ul>
  <li>Iris-setosa</li>
  <li>Iris-versicolor</li>
  <li>Iris-virginica</li>
</ul>

<p>
The dataset includes the following features:
</p>
<ul>
  <li>Sepal Length</li>
  <li>Sepal Width</li>
  <li>Petal Length</li>
  <li>Petal Width</li>
</ul>

---

<h2>2. Data Preprocessing</h2>
<p>
The <b>Id</b> column is removed since it does not help in prediction.
</p>

<p>
The target column (<b>Species</b>) is converted into numerical values so that the
machine learning model can process it.
</p>

---

<h2>3. Feature and Target Separation</h2>
<p>
The dataset is divided into:
</p>
<ul>
  <li><b>X (Features)</b>: Flower measurements</li>
  <li><b>y (Target)</b>: Encoded flower species</li>
</ul>

---

<h2>4. Train-Test Split</h2>
<p>
The dataset is split into:
</p>
<ul>
  <li>80% Training Data</li>
  <li>20% Testing Data</li>
</ul>

<p>
This helps evaluate the model on unseen data.
</p>

---

<h2>5. Model Training</h2>
<p>
A <b>Decision Tree Classifier</b> from <b>scikit-learn</b> is used to train the
model on the training dataset.
</p>

<p>
During training, the model:
</p>
<ul>
  <li>Selects optimal features for splitting</li>
  <li>Builds a tree structure based on decision rules</li>
  <li>Stops splitting when conditions are satisfied</li>
</ul>

---

<h2>6. Model Evaluation</h2>
<p>
Model performance is evaluated using <b>accuracy score</b> on both:
</p>
<ul>
  <li>Training Data</li>
  <li>Testing Data</li>
</ul>

<p>
Good test accuracy indicates effective learning without overfitting.
</p>

---

<h2>7. Prediction with Custom Input</h2>
<p>
The model allows predictions using custom flower measurements.
</p>

<p>
Based on the predicted class:
</p>
<ul>
  <li>0 → Setosa</li>
  <li>1 → Versicolor</li>
  <li>2 → Virginica</li>
</ul>

---

<h2>8. Advantages of Decision Tree</h2>
<ul>
  <li>Simple and intuitive model</li>
  <li>Handles non-linear data</li>
  <li>No feature scaling required</li>
</ul>

---

<h2>9. Conclusion</h2>
<p>
This project demonstrates the implementation of a Decision Tree classifier for
multi-class classification. The model effectively learns decision rules from the
data and provides accurate predictions, making it suitable for real-world
classification tasks.
</p>

<h1>🌳 Random Forest – Detailed Explanation</h1>

<h2>1. What is Random Forest?</h2>
<p>
<b>Random Forest</b> is an ensemble machine learning algorithm that builds
multiple decision trees and combines their predictions to produce a more
accurate and stable result.
</p>
<p>
It can be used for:
</p>
<ul>
  <li><b>Classification</b> (e.g., spam detection)</li>
  <li><b>Regression</b> (e.g., house price prediction)</li>
</ul>
<p>
<b>Key Idea:</b> Many weak models together form a strong model.
</p>

<hr>

<h2>2. Why Do We Need Random Forest?</h2>

<h3>Problems with Decision Trees</h3>
<ul>
  <li>Prone to overfitting</li>
  <li>High variance</li>
  <li>Small data changes can drastically change the tree</li>
</ul>

<h3>How Random Forest Solves This</h3>
<ul>
  <li>Reduces overfitting</li>
  <li>Improves accuracy</li>
  <li>Creates a more stable model</li>
</ul>

<hr>

<h2>3. How Random Forest Works (Step-by-Step)</h2>

<h3>Step 1: Bootstrap Sampling (Bagging)</h3>
<p>
Random samples are drawn from the original dataset <b>with replacement</b>.
Each decision tree is trained on a different sample.
</p>
<ul>
  <li>Some records are repeated</li>
  <li>Some records are left out</li>
</ul>

<h3>Step 2: Feature Randomness</h3>
<p>
At each split in a tree, only a random subset of features is considered.
This prevents dominant features from controlling all trees.
</p>

<h3>Step 3: Build Decision Trees</h3>
<ul>
  <li>Each tree is built independently</li>
  <li>Trees are usually deep and unpruned</li>
</ul>

<h3>Step 4: Combine Predictions</h3>

<h4>For Classification</h4>
<p>
Each tree votes for a class. The class with the majority votes becomes the final prediction.
</p>

<h4>For Regression</h4>
<p>
The final prediction is the average of all tree outputs.
</p>

<hr>

<h2>4. Mathematical Intuition</h2>
<p>
Random Forest reduces variance by averaging multiple independent models:
</p>
<p>
<b>Variance of Average = Variance of Tree / Number of Trees</b>
</p>
<p>
More trees and lower correlation between them lead to better performance.
</p>

<hr>

<h2>5. Important Hyperparameters</h2>

<h3>n_estimators</h3>
<p>Number of decision trees in the forest.</p>

<h3>max_depth</h3>
<p>Maximum depth of each tree. Controls overfitting.</p>

<h3>min_samples_split</h3>
<p>Minimum number of samples required to split a node.</p>

<h3>min_samples_leaf</h3>
<p>Minimum number of samples required in a leaf node.</p>

<h3>max_features</h3>
<p>
Number of features considered at each split.
</p>
<ul>
  <li>Classification: sqrt(number of features)</li>
  <li>Regression: all features</li>
</ul>

<h3>criterion</h3>
<ul>
  <li>Classification: Gini, Entropy</li>
  <li>Regression: Mean Squared Error, Mean Absolute Error</li>
</ul>

<hr>

<h2>6. Advantages of Random Forest</h2>
<ul>
  <li>Reduces overfitting</li>
  <li>High accuracy</li>
  <li>Works well with large datasets</li>
  <li>Handles missing values</li>
  <li>Provides feature importance</li>
</ul>

<hr>

<h2>7. Disadvantages of Random Forest</h2>
<ul>
  <li>Slower than a single decision tree</li>
  <li>Harder to interpret</li>
  <li>Uses more memory</li>
</ul>

<hr>

<h2>8. Feature Importance</h2>
<p>
Random Forest provides feature importance scores based on how much each feature
reduces impurity across all trees. This helps in feature selection and model interpretation.
</p>

<hr>

<h2>9. Random Forest vs Decision Tree</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>Aspect</th>
    <th>Decision Tree</th>
    <th>Random Forest</th>
  </tr>
  <tr>
    <td>Overfitting</td>
    <td>High</td>
    <td>Low</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>Medium</td>
    <td>High</td>
  </tr>
  <tr>
    <td>Stability</td>
    <td>Low</td>
    <td>High</td>
  </tr>
  <tr>
    <td>Interpretability</td>
    <td>High</td>
    <td>Low</td>
  </tr>
</table>

<hr>

<h2>10. When to Use Random Forest</h2>

<h3>Recommended</h3>
<ul>
  <li>Tabular data</li>
  <li>Medium to large datasets</li>
  <li>Non-linear relationships</li>
</ul>

<h3>Not Recommended</h3>
<ul>
  <li>Very small datasets</li>
  <li>Real-time low-latency systems</li>
  <li>When model size is critical</li>
</ul>

<hr>

<h2>11. Summary</h2>
<p>
<b>Random Forest</b> combines multiple deep decision trees trained on random data
samples and random feature subsets to produce a powerful, accurate, and robust model.
</p>



<h1>🚀 AdaBoost (Adaptive Boosting) – Detailed Explanation</h1>

<h2>1. What is AdaBoost?</h2>
<p>
<b>AdaBoost (Adaptive Boosting)</b> is an ensemble learning algorithm that
combines multiple <b>weak learners</b> to create a strong predictive model.
</p>
<p>
It works by training models sequentially, where each new model focuses more on
the mistakes made by the previous ones.
</p>

<p>
AdaBoost is mainly used for:
</p>
<ul>
  <li><b>Classification</b></li>
  <li><b>Regression</b> (AdaBoost Regressor)</li>
</ul>

<hr>

<h2>2. Why AdaBoost?</h2>

<h3>Problems with Single Models</h3>
<ul>
  <li>Low accuracy</li>
  <li>High bias</li>
  <li>Fails to capture complex patterns</li>
</ul>

<h3>How AdaBoost Solves This</h3>
<ul>
  <li>Focuses on difficult data points</li>
  <li>Improves accuracy iteratively</li>
  <li>Turns weak learners into a strong learner</li>
</ul>

<hr>

<h2>3. Key Idea Behind AdaBoost</h2>
<p>
AdaBoost gives <b>more importance (weight)</b> to incorrectly classified data points
and less importance to correctly classified ones.
</p>
<p>
Each new weak learner tries harder to classify the previously misclassified samples.
</p>

<hr>

<h2>4. How AdaBoost Works (Step-by-Step)</h2>

<h3>Step 1: Initialize Weights</h3>
<p>
All training samples are given equal weights initially.
</p>

<h3>Step 2: Train a Weak Learner</h3>
<p>
A weak learner (usually a decision stump – a tree with depth 1) is trained on the data.
</p>

<h3>Step 3: Calculate Error</h3>
<p>
The error is calculated based on misclassified samples and their weights.
</p>

<h3>Step 4: Assign Model Weight</h3>
<p>
Each weak learner is assigned a weight based on its accuracy.
Better models get higher weight.
</p>

<h3>Step 5: Update Sample Weights</h3>
<ul>
  <li>Weights of misclassified samples are increased</li>
  <li>Weights of correctly classified samples are decreased</li>
</ul>

<h3>Step 6: Repeat</h3>
<p>
Steps 2–5 are repeated until the required number of models is trained.
</p>

<h3>Step 7: Final Prediction</h3>
<p>
The final output is a <b>weighted combination</b> of all weak learners.
</p>

<hr>

<h2>5. Mathematical Intuition (Simple)</h2>

<p>
Error of model:
</p>
<p>
<b>Error = Sum of weights of misclassified samples</b>
</p>

<p>
Model weight:
</p>
<p>
<b>α = ½ × ln((1 − error) / error)</b>
</p>

<p>
Final prediction:
</p>
<p>
<b>Sign of weighted sum of all weak learners</b>
</p>

<hr>

<h2>6. Weak Learners in AdaBoost</h2>
<p>
AdaBoost commonly uses <b>Decision Stumps</b>:
</p>
<ul>
  <li>Single split decision trees</li>
  <li>Very simple models</li>
  <li>Slightly better than random guessing</li>
</ul>

<hr>

<h2>7. Important Hyperparameters</h2>

<h3>n_estimators</h3>
<p>
Number of weak learners to train.
</p>

<h3>learning_rate</h3>
<p>
Controls how much each weak learner contributes to the final model.
Lower values make learning slower but more robust.
</p>

<h3>base_estimator</h3>
<p>
The weak learner used by AdaBoost (default is a decision stump).
</p>

<hr>

<h2>8. Advantages of AdaBoost</h2>
<ul>
  <li>High accuracy</li>
  <li>Reduces bias effectively</li>
  <li>Easy to implement</li>
  <li>Works well with simple models</li>
</ul>

<hr>

<h2>9. Disadvantages of AdaBoost</h2>
<ul>
  <li>Sensitive to noisy data</li>
  <li>Performance degrades with outliers</li>
  <li>Sequential training makes it slower</li>
</ul>

<hr>

<h2>10. AdaBoost vs Random Forest</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>Aspect</th>
    <th>AdaBoost</th>
    <th>Random Forest</th>
  </tr>
  <tr>
    <td>Learning Style</td>
    <td>Sequential</td>
    <td>Parallel</td>
  </tr>
  <tr>
    <td>Focus</td>
    <td>Hard samples</td>
    <td>Overall variance reduction</td>
  </tr>
  <tr>
    <td>Noise Sensitivity</td>
    <td>High</td>
    <td>Low</td>
  </tr>
  <tr>
    <td>Overfitting</td>
    <td>Low (with clean data)</td>
    <td>Very Low</td>
  </tr>
</table>

<hr>

<h2>11. When to Use AdaBoost</h2>

<h3>Recommended</h3>
<ul>
  <li>Clean datasets</li>
  <li>Binary classification problems</li>
  <li>When base models are weak</li>
</ul>

<h3>Not Recommended</h3>
<ul>
  <li>Highly noisy data</li>
  <li>Datasets with many outliers</li>
</ul>

<hr>

<h2>12. Summary</h2>
<p>
<b>AdaBoost</b> builds a strong model by combining multiple weak learners,
giving more importance to difficult samples and improving predictions step by step.
</p>


<h1>📈 Gradient Boosting – Detailed Explanation</h1>

<h2>1. What is Gradient Boosting?</h2>
<p>
<b>Gradient Boosting</b> is an ensemble learning algorithm that builds models
<b>sequentially</b>, where each new model tries to correct the errors made by
the previous models.
</p>
<p>
Unlike Random Forest (parallel trees), Gradient Boosting focuses on
<b>reducing errors step by step</b>.
</p>

<p>
It is used for:
</p>
<ul>
  <li><b>Classification</b></li>
  <li><b>Regression</b></li>
</ul>

<hr>

<h2>2. Why Gradient Boosting?</h2>

<h3>Limitations of Simple Models</h3>
<ul>
  <li>High bias</li>
  <li>Cannot capture complex patterns</li>
</ul>

<h3>How Gradient Boosting Helps</h3>
<ul>
  <li>Reduces bias gradually</li>
  <li>Learns complex non-linear relationships</li>
  <li>Improves accuracy iteratively</li>
</ul>

<hr>

<h2>3. Core Idea of Gradient Boosting</h2>
<p>
Gradient Boosting trains new models to predict the <b>residual errors</b>
of previous models.
</p>

<p>
Residual = Actual Value − Predicted Value
</p>

<p>
Each new model focuses on minimizing a <b>loss function</b>
using gradient descent.
</p>

<hr>

<h2>4. How Gradient Boosting Works (Step-by-Step)</h2>

<h3>Step 1: Initialize Model</h3>
<p>
Start with a simple prediction (mean value for regression or log-odds for classification).
</p>

<h3>Step 2: Compute Residuals</h3>
<p>
Calculate the difference between actual values and predicted values.
</p>

<h3>Step 3: Train a Weak Learner</h3>
<p>
A weak learner (usually a shallow decision tree) is trained on the residuals.
</p>

<h3>Step 4: Update Predictions</h3>
<p>
Predictions are updated by adding the new model’s output multiplied by a learning rate.
</p>

<h3>Step 5: Repeat</h3>
<p>
Steps 2–4 are repeated until the specified number of models is reached.
</p>

<h3>Step 6: Final Prediction</h3>
<p>
The final prediction is the sum of predictions from all weak learners.
</p>

<hr>

<h2>5. Loss Function and Gradient Descent</h2>
<p>
Gradient Boosting minimizes a loss function using gradient descent.
</p>

<ul>
  <li>Regression: Mean Squared Error (MSE)</li>
  <li>Classification: Log Loss</li>
</ul>

<p>
Each model moves the prediction in the direction that reduces the loss.
</p>

<hr>

<h2>6. Weak Learners in Gradient Boosting</h2>
<ul>
  <li>Shallow decision trees</li>
  <li>Low depth (usually depth = 3)</li>
  <li>Low variance, high bias models</li>
</ul>

<hr>

<h2>7. Important Hyperparameters</h2>

<h3>n_estimators</h3>
<p>
Number of boosting stages (trees).
</p>

<h3>learning_rate</h3>
<p>
Controls how much each tree contributes to the final prediction.
Smaller values require more trees but improve generalization.
</p>

<h3>max_depth</h3>
<p>
Maximum depth of individual trees.
</p>

<h3>subsample</h3>
<p>
Fraction of samples used for training each tree.
Helps reduce overfitting.
</p>

<hr>

<h2>8. Advantages of Gradient Boosting</h2>
<ul>
  <li>High predictive accuracy</li>
  <li>Handles complex patterns</li>
  <li>Works well with structured data</li>
  <li>Flexible loss functions</li>
</ul>

<hr>

<h2>9. Disadvantages of Gradient Boosting</h2>
<ul>
  <li>Sensitive to noisy data</li>
  <li>Can overfit if not tuned properly</li>
  <li>Slower training due to sequential learning</li>
</ul>

<hr>

<h2>10. Gradient Boosting vs AdaBoost</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>Aspect</th>
    <th>Gradient Boosting</th>
    <th>AdaBoost</th>
  </tr>
  <tr>
    <td>Error Handling</td>
    <td>Uses gradients of loss function</td>
    <td>Reweights misclassified samples</td>
  </tr>
  <tr>
    <td>Loss Function</td>
    <td>Customizable</td>
    <td>Exponential loss</td>
  </tr>
  <tr>
    <td>Noise Sensitivity</td>
    <td>Medium</td>
    <td>High</td>
  </tr>
  <tr>
    <td>Flexibility</td>
    <td>High</td>
    <td>Low</td>
  </tr>
</table>

<hr>

<h2>11. When to Use Gradient Boosting</h2>

<h3>Recommended</h3>
<ul>
  <li>Medium to large datasets</li>
  <li>Complex relationships</li>
  <li>When high accuracy is required</li>
</ul>

<h3>Not Recommended</h3>
<ul>
  <li>Very noisy datasets</li>
  <li>Real-time low-latency applications</li>
</ul>

<hr>

<h2>12. Summary</h2>
<p>
<b>Gradient Boosting</b> builds strong models by sequentially adding weak learners
that correct previous errors using gradient descent, resulting in highly accurate predictions.
</p>


<h1>⚡ XGBoost (Extreme Gradient Boosting) – Detailed Explanation</h1>

<h2>1. What is XGBoost?</h2>
<p>
<b>XGBoost (Extreme Gradient Boosting)</b> is an advanced and optimized
implementation of the Gradient Boosting algorithm designed for
<b>high performance, speed, and accuracy</b>.
</p>

<p>
XGBoost is widely used in:
</p>
<ul>
  <li>Machine learning competitions</li>
  <li>Industry-scale predictive systems</li>
  <li>Both classification and regression tasks</li>
</ul>

<hr>

<h2>2. Why XGBoost?</h2>

<h3>Limitations of Traditional Gradient Boosting</h3>
<ul>
  <li>Slow training</li>
  <li>Overfitting risk</li>
  <li>High memory usage</li>
</ul>

<h3>How XGBoost Improves Gradient Boosting</h3>
<ul>
  <li>Optimized for speed and performance</li>
  <li>Built-in regularization</li>
  <li>Efficient handling of missing values</li>
  <li>Parallel tree construction</li>
</ul>

<hr>

<h2>3. Core Idea of XGBoost</h2>
<p>
XGBoost builds trees sequentially like Gradient Boosting, but it minimizes
a <b>regularized objective function</b> that includes:
</p>

<ul>
  <li>Loss function</li>
  <li>Regularization term (L1 & L2)</li>
</ul>

<p>
This helps control model complexity and prevent overfitting.
</p>

<hr>

<h2>4. How XGBoost Works (Step-by-Step)</h2>

<h3>Step 1: Initialize Predictions</h3>
<p>
Start with an initial prediction (mean for regression or log-odds for classification).
</p>

<h3>Step 2: Compute Gradients & Hessians</h3>
<p>
XGBoost uses both first-order (gradient) and second-order (Hessian) derivatives
of the loss function.
</p>

<h3>Step 3: Build Trees</h3>
<p>
Trees are grown by choosing splits that maximize the reduction in loss.
</p>

<h3>Step 4: Apply Regularization</h3>
<p>
Tree complexity is penalized using L1 and L2 regularization.
</p>

<h3>Step 5: Update Predictions</h3>
<p>
Predictions are updated using a learning rate.
</p>

<h3>Step 6: Repeat</h3>
<p>
Steps 2–5 are repeated until the specified number of trees is reached.
</p>

<hr>

<h2>5. Objective Function</h2>
<p>
XGBoost minimizes the following objective:
</p>

<p>
<b>Objective = Loss Function + Regularization</b>
</p>

<ul>
  <li>Loss measures prediction error</li>
  <li>Regularization controls model complexity</li>
</ul>

<hr>

<h2>6. Handling Missing Values</h2>
<p>
XGBoost automatically learns the best direction to handle missing values
during tree splitting.
</p>

<p>
No explicit imputation is required.
</p>

<hr>

<h2>7. Important Hyperparameters</h2>

<h3>n_estimators</h3>
<p>Number of trees.</p>

<h3>learning_rate (eta)</h3>
<p>Controls step size during optimization.</p>

<h3>max_depth</h3>
<p>Maximum depth of each tree.</p>

<h3>subsample</h3>
<p>Fraction of samples used for each tree.</p>

<h3>colsample_bytree</h3>
<p>Fraction of features used per tree.</p>

<h3>reg_alpha (L1)</h3>
<p>L1 regularization term.</p>

<h3>reg_lambda (L2)</h3>
<p>L2 regularization term.</p>

<hr>

<h2>8. Advantages of XGBoost</h2>
<ul>
  <li>Very high predictive accuracy</li>
  <li>Fast training and prediction</li>
  <li>Built-in regularization</li>
  <li>Handles missing values</li>
  <li>Supports parallel processing</li>
</ul>

<hr>

<h2>9. Disadvantages of XGBoost</h2>
<ul>
  <li>Complex hyperparameter tuning</li>
  <li>Less interpretable</li>
  <li>Higher memory usage</li>
</ul>

<hr>

<h2>10. XGBoost vs Gradient Boosting</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>Aspect</th>
    <th>XGBoost</th>
    <th>Gradient Boosting</th>
  </tr>
  <tr>
    <td>Speed</td>
    <td>Very Fast</td>
    <td>Slower</td>
  </tr>
  <tr>
    <td>Regularization</td>
    <td>Yes (L1 & L2)</td>
    <td>No (default)</td>
  </tr>
  <tr>
    <td>Missing Values</td>
    <td>Handled Automatically</td>
    <td>Needs Preprocessing</td>
  </tr>
  <tr>
    <td>Parallelization</td>
    <td>Yes</td>
    <td>Limited</td>
  </tr>
</table>

<hr>

<h2>11. When to Use XGBoost</h2>

<h3>Recommended</h3>
<ul>
  <li>Structured/tabular datasets</li>
  <li>Large datasets</li>
  <li>High-performance requirements</li>
</ul>

<h3>Not Recommended</h3>
<ul>
  <li>Very small datasets</li>
  <li>Simple problems</li>
</ul>

<hr>

<h2>12. Summary</h2>
<p>
<b>XGBoost</b> is a powerful and efficient boosting algorithm that improves
Gradient Boosting by adding regularization, parallelization, and smart handling
of missing values, making it one of the most successful ML algorithms in practice.
</p>


<h1>📌 Support Vector Machine (SVM) – Detailed Explanation</h1>

<h2>1. What is Support Vector Machine?</h2>
<p>
<b>Support Vector Machine (SVM)</b> is a supervised machine learning algorithm
used for <b>classification</b> and <b>regression</b> problems.
</p>
<p>
The main objective of SVM is to find an <b>optimal decision boundary</b>
(called a hyperplane) that best separates data points of different classes.
</p>

<hr>

<h2>2. Why SVM?</h2>

<h3>Problems with Traditional Algorithms</h3>
<ul>
  <li>Fail to handle high-dimensional data</li>
  <li>Poor performance with complex boundaries</li>
</ul>

<h3>How SVM Solves This</h3>
<ul>
  <li>Works well in high-dimensional spaces</li>
  <li>Handles non-linear boundaries using kernels</li>
  <li>Focuses on the most important data points</li>
</ul>

<hr>

<h2>3. Core Idea of SVM</h2>
<p>
SVM finds a hyperplane that maximizes the <b>margin</b>
(distance between the hyperplane and the nearest data points).
</p>

<p>
The nearest data points are called <b>Support Vectors</b>.
</p>

<p>
Larger margin → Better generalization.
</p>

<hr>

<h2>4. What is a Hyperplane?</h2>
<p>
A hyperplane is a decision boundary that separates data points.
</p>

<ul>
  <li>In 2D: a line</li>
  <li>In 3D: a plane</li>
  <li>In higher dimensions: a hyperplane</li>
</ul>

<hr>

<h2>5. Support Vectors</h2>
<p>
Support vectors are the data points closest to the hyperplane.
</p>
<ul>
  <li>They define the position of the hyperplane</li>
  <li>Removing other points does not affect the boundary</li>
</ul>

<hr>

<h2>6. Types of SVM</h2>

<h3>Linear SVM</h3>
<p>
Used when data is linearly separable.
</p>

<h3>Non-Linear SVM</h3>
<p>
Used when data is not linearly separable.
Uses <b>kernel trick</b>.
</p>

<hr>

<h2>7. Kernel Trick</h2>
<p>
The kernel trick transforms data into a higher-dimensional space
where it becomes linearly separable.
</p>

<h3>Common Kernels</h3>
<ul>
  <li><b>Linear</b> – For linearly separable data</li>
  <li><b>Polynomial</b> – Captures polynomial relationships</li>
  <li><b>RBF (Gaussian)</b> – Handles complex non-linear patterns</li>
  <li><b>Sigmoid</b> – Similar to neural networks</li>
</ul>

<hr>

<h2>8. How SVM Works (Step-by-Step)</h2>

<h3>Step 1: Choose a Kernel</h3>
<p>
Select a kernel function based on data complexity.
</p>

<h3>Step 2: Find Optimal Hyperplane</h3>
<p>
SVM finds the hyperplane with the maximum margin.
</p>

<h3>Step 3: Identify Support Vectors</h3>
<p>
Only support vectors influence the decision boundary.
</p>

<h3>Step 4: Make Predictions</h3>
<p>
New data points are classified based on which side of the hyperplane they fall.
</p>

<hr>

<h2>9. Important Hyperparameters</h2>

<h3>C (Regularization Parameter)</h3>
<p>
Controls the trade-off between maximizing margin and minimizing classification error.
</p>
<ul>
  <li>Small C → Larger margin, more misclassification</li>
  <li>Large C → Smaller margin, less misclassification</li>
</ul>

<h3>kernel</h3>
<p>
Specifies the kernel type (linear, poly, rbf, sigmoid).
</p>

<h3>gamma</h3>
<p>
Defines the influence of a single training example.
</p>
<ul>
  <li>Low gamma → Far influence</li>
  <li>High gamma → Near influence</li>
</ul>

<h3>degree</h3>
<p>
Degree of polynomial kernel.
</p>

<hr>

<h2>10. Advantages of SVM</h2>
<ul>
  <li>Effective in high-dimensional spaces</li>
  <li>Memory efficient</li>
  <li>Works well with clear margin separation</li>
  <li>Robust to overfitting</li>
</ul>

<hr>

<h2>11. Disadvantages of SVM</h2>
<ul>
  <li>Slow for very large datasets</li>
  <li>Choosing kernel and parameters is difficult</li>
  <li>Less interpretable</li>
</ul>

<hr>

<h2>12. SVM vs Logistic Regression</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>Aspect</th>
    <th>SVM</th>
    <th>Logistic Regression</th>
  </tr>
  <tr>
    <td>Decision Boundary</td>
    <td>Maximum margin</td>
    <td>Probability-based</td>
  </tr>
  <tr>
    <td>Non-linearity</td>
    <td>Handled via kernels</td>
    <td>Needs feature engineering</td>
  </tr>
  <tr>
    <td>Performance</td>
    <td>High (small/medium data)</td>
    <td>Good (large data)</td>
  </tr>
</table>

<hr>

<h2>13. When to Use SVM</h2>

<h3>Recommended</h3>
<ul>
  <li>High-dimensional datasets</li>
  <li>Text classification</li>
  <li>Image classification (small to medium data)</li>
</ul>

<h3>Not Recommended</h3>
<ul>
  <li>Very large datasets</li>
  <li>Highly noisy data</li>
</ul>

<hr>

<h2>14. Summary</h2>
<p>
<b>Support Vector Machine</b> finds the optimal hyperplane that maximizes
the margin between classes, using support vectors and kernel functions
to achieve strong performance on both linear and non-linear data.
</p>


<h1>📊 ROC Curve & AUC – Detailed Explanation</h1>

<h2>1. What is ROC Curve?</h2>
<p>
<b>ROC (Receiver Operating Characteristic) Curve</b> is a performance evaluation
metric used for <b>binary classification models</b>.
</p>
<p>
It shows the trade-off between:
</p>
<ul>
  <li><b>True Positive Rate (TPR)</b></li>
  <li><b>False Positive Rate (FPR)</b></li>
</ul>
<p>
at different classification threshold values.
</p>

<hr>

<h2>2. Why ROC Curve?</h2>

<h3>Limitations of Accuracy</h3>
<ul>
  <li>Accuracy can be misleading for imbalanced datasets</li>
  <li>Does not consider probability thresholds</li>
</ul>

<h3>Benefits of ROC Curve</h3>
<ul>
  <li>Evaluates model performance across all thresholds</li>
  <li>Independent of class distribution</li>
  <li>Helps compare multiple classifiers</li>
</ul>

<hr>

<h2>3. Confusion Matrix Basics</h2>

<table border="1" cellpadding="8">
  <tr>
    <th></th>
    <th>Predicted Positive</th>
    <th>Predicted Negative</th>
  </tr>
  <tr>
    <th>Actual Positive</th>
    <td>True Positive (TP)</td>
    <td>False Negative (FN)</td>
  </tr>
  <tr>
    <th>Actual Negative</th>
    <td>False Positive (FP)</td>
    <td>True Negative (TN)</td>
  </tr>
</table>

<hr>

<h2>4. Key Terms Used in ROC</h2>

<h3>True Positive Rate (TPR) / Recall</h3>
<p>
TPR = TP / (TP + FN)
</p>

<h3>False Positive Rate (FPR)</h3>
<p>
FPR = FP / (FP + TN)
</p>

<hr>

<h2>5. How ROC Curve is Drawn</h2>
<ul>
  <li>Model outputs probabilities</li>
  <li>Different threshold values are applied</li>
  <li>TPR and FPR are calculated for each threshold</li>
  <li>TPR is plotted against FPR</li>
</ul>

<hr>

<h2>6. What is AUC?</h2>
<p>
<b>AUC (Area Under the Curve)</b> represents the area under the ROC curve.
</p>
<p>
It measures the model’s ability to distinguish between positive and negative classes.
</p>

<hr>

<h2>7. AUC Score Interpretation</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>AUC Value</th>
    <th>Interpretation</th>
  </tr>
  <tr>
    <td>1.0</td>
    <td>Perfect classifier</td>
  </tr>
  <tr>
    <td>0.9 – 1.0</td>
    <td>Excellent</td>
  </tr>
  <tr>
    <td>0.8 – 0.9</td>
    <td>Good</td>
  </tr>
  <tr>
    <td>0.7 – 0.8</td>
    <td>Fair</td>
  </tr>
  <tr>
    <td>0.5</td>
    <td>Random guessing</td>
  </tr>
  <tr>
    <td>&lt; 0.5</td>
    <td>Worse than random</td>
  </tr>
</table>

<hr>

<h2>8. ROC Curve Shape Interpretation</h2>
<ul>
  <li>Closer to top-left corner → Better model</li>
  <li>Diagonal line → Random classifier</li>
  <li>Curve below diagonal → Poor model</li>
</ul>

<hr>

<h2>9. ROC Curve vs Precision-Recall Curve</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>Aspect</th>
    <th>ROC Curve</th>
    <th>Precision-Recall Curve</th>
  </tr>
  <tr>
    <td>Focus</td>
    <td>TPR vs FPR</td>
    <td>Precision vs Recall</td>
  </tr>
  <tr>
    <td>Best For</td>
    <td>Balanced datasets</td>
    <td>Highly imbalanced datasets</td>
  </tr>
</table>

<hr>

<h2>10. Advantages of ROC & AUC</h2>
<ul>
  <li>Threshold-independent evaluation</li>
  <li>Good for model comparison</li>
  <li>Works well with imbalanced data</li>
</ul>

<hr>

<h2>11. Limitations of ROC & AUC</h2>
<ul>
  <li>Not ideal when false positives are very costly</li>
  <li>AUC does not give probability calibration</li>
</ul>

<hr>

<h2>12. When to Use ROC & AUC</h2>
<ul>
  <li>Binary classification problems</li>
  <li>Model comparison</li>
  <li>Imbalanced datasets</li>
</ul>

<hr>

<h2>13. Summary</h2>
<p>
<b>ROC Curve</b> visualizes the trade-off between True Positive Rate and
False Positive Rate across different thresholds, while <b>AUC</b> summarizes
this performance into a single score representing how well the model
distinguishes between classes.
</p>

<h1>🔁 Cross Validation – Detailed Explanation</h1>

<h2>1. What is Cross Validation?</h2>
<p>
<b>Cross Validation</b> is a model evaluation technique used to assess how well
a machine learning model generalizes to unseen data.
</p>
<p>
Instead of using a single train-test split, cross validation repeatedly
trains and tests the model on different subsets of the data.
</p>

<hr>

<h2>2. Why Cross Validation?</h2>

<h3>Problems with Single Train-Test Split</h3>
<ul>
  <li>Model performance depends on how data is split</li>
  <li>High variance in evaluation results</li>
  <li>Risk of overfitting or underfitting</li>
</ul>

<h3>Benefits of Cross Validation</h3>
<ul>
  <li>More reliable performance estimate</li>
  <li>Uses entire dataset effectively</li>
  <li>Reduces evaluation bias</li>
</ul>

<hr>

<h2>3. How Cross Validation Works</h2>
<p>
The dataset is divided into multiple parts called <b>folds</b>.
</p>
<ul>
  <li>One fold is used for testing</li>
  <li>Remaining folds are used for training</li>
  <li>The process is repeated for each fold</li>
</ul>

<p>
Final performance is the <b>average score</b> across all folds.
</p>

<hr>

<h2>4. K-Fold Cross Validation</h2>
<p>
<b>K-Fold Cross Validation</b> is the most commonly used technique.
</p>

<ul>
  <li>Dataset is split into <b>K</b> equal folds</li>
  <li>Model is trained K times</li>
  <li>Each fold is used once as test data</li>
</ul>

<p>
Example:
</p>
<ul>
  <li>K = 5 → Train 5 times</li>
  <li>K = 10 → Train 10 times</li>
</ul>

<hr>

<h2>5. Stratified K-Fold Cross Validation</h2>
<p>
<b>Stratified K-Fold</b> ensures that each fold has the same class distribution
as the original dataset.
</p>
<p>
It is especially useful for <b>imbalanced datasets</b>.
</p>

<hr>

<h2>6. Leave-One-Out Cross Validation (LOOCV)</h2>
<p>
<b>Leave-One-Out Cross Validation</b> uses:
</p>
<ul>
  <li>1 sample for testing</li>
  <li>Remaining samples for training</li>
</ul>
<p>
This process is repeated for every data point.
</p>

<p>
Advantages:
</p>
<ul>
  <li>Very low bias</li>
</ul>

<p>
Disadvantages:
</p>
<ul>
  <li>Very high computational cost</li>
</ul>

<hr>

<h2>7. Time Series Cross Validation</h2>
<p>
Used when data has a time dependency.
</p>
<ul>
  <li>Training data always comes before test data</li>
  <li>Future data is never used to predict past data</li>
</ul>

<hr>

<h2>8. Cross Validation vs Train-Test Split</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>Aspect</th>
    <th>Train-Test Split</th>
    <th>Cross Validation</th>
  </tr>
  <tr>
    <td>Data Usage</td>
    <td>Partial</td>
    <td>Full</td>
  </tr>
  <tr>
    <td>Performance Estimate</td>
    <td>Unstable</td>
    <td>Stable</td>
  </tr>
  <tr>
    <td>Computation</td>
    <td>Fast</td>
    <td>Slower</td>
  </tr>
</table>

<hr>

<h2>9. Advantages of Cross Validation</h2>
<ul>
  <li>Better generalization estimation</li>
  <li>Reduces overfitting risk</li>
  <li>Works well with small datasets</li>
</ul>

<hr>

<h2>10. Disadvantages of Cross Validation</h2>
<ul>
  <li>Computationally expensive</li>
  <li>Not suitable for very large datasets</li>
</ul>

<hr>

<h2>11. When to Use Cross Validation</h2>
<ul>
  <li>Small to medium datasets</li>
  <li>Model selection</li>
  <li>Hyperparameter tuning</li>
</ul>

<hr>

<h2>12. Cross Validation in Hyperparameter Tuning</h2>
<p>
Cross validation is commonly used with:
</p>
<ul>
  <li>GridSearchCV</li>
  <li>RandomizedSearchCV</li>
</ul>

<p>
This ensures the selected hyperparameters generalize well.
</p>

<hr>

<h2>13. Summary</h2>
<p>
<b>Cross Validation</b> evaluates model performance by repeatedly training and
testing on different data splits, providing a more reliable and unbiased
estimate of how the model will perform on unseen data.
</p>


<h1>⚙️ Hyperparameter Tuning – Detailed Explanation</h1>

<h2>1. What is Hyperparameter Tuning?</h2>
<p>
<b>Hyperparameter Tuning</b> is the process of finding the best set of
<b>hyperparameters</b> for a machine learning model to achieve optimal performance.
</p>
<p>
Hyperparameters are parameters that are set <b>before training</b> and control
how the model learns.
</p>

<hr>

<h2>2. Parameters vs Hyperparameters</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>Aspect</th>
    <th>Parameters</th>
    <th>Hyperparameters</th>
  </tr>
  <tr>
    <td>Definition</td>
    <td>Learned from data</td>
    <td>Set before training</td>
  </tr>
  <tr>
    <td>Examples</td>
    <td>Weights, coefficients</td>
    <td>Learning rate, max depth, C</td>
  </tr>
</table>

<hr>

<h2>3. Why Hyperparameter Tuning?</h2>

<h3>Problems Without Tuning</h3>
<ul>
  <li>Underfitting or overfitting</li>
  <li>Poor generalization</li>
  <li>Suboptimal performance</li>
</ul>

<h3>Benefits of Tuning</h3>
<ul>
  <li>Improves model accuracy</li>
  <li>Better bias-variance tradeoff</li>
  <li>Enhances model robustness</li>
</ul>

<hr>

<h2>4. Common Hyperparameters</h2>

<ul>
  <li>Learning rate</li>
  <li>Number of estimators</li>
  <li>Maximum depth</li>
  <li>Regularization parameters</li>
  <li>Kernel parameters</li>
</ul>

<hr>

<h2>5. Hyperparameter Tuning Techniques</h2>

<h3>1. Manual Search</h3>
<p>
Manually selecting hyperparameters based on experience.
</p>
<p>
<b>Disadvantage:</b> Time-consuming and not scalable.
</p>

<hr>

<h3>2. Grid Search</h3>
<p>
Grid Search evaluates all possible combinations of given hyperparameters.
</p>

<p>
Advantages:
</p>
<ul>
  <li>Systematic</li>
  <li>Easy to implement</li>
</ul>

<p>
Disadvantages:
</p>
<ul>
  <li>Computationally expensive</li>
</ul>

<hr>

<h3>3. Random Search</h3>
<p>
Random Search selects random combinations of hyperparameters.
</p>

<p>
Advantages:
</p>
<ul>
  <li>Faster than grid search</li>
  <li>Explores larger space</li>
</ul>

<p>
Disadvantages:
</p>
<ul>
  <li>May miss optimal values</li>
</ul>

<hr>

<h3>4. Bayesian Optimization</h3>
<p>
Uses probabilistic models to select the next best hyperparameters.
</p>

<p>
Advantages:
</p>
<ul>
  <li>Efficient</li>
  <li>Fewer evaluations</li>
</ul>

<hr>

<h2>6. Hyperparameter Tuning with Cross Validation</h2>
<p>
Hyperparameter tuning is usually combined with <b>cross validation</b>
to ensure robust performance estimation.
</p>

<p>
This helps avoid overfitting to a single train-test split.
</p>

<hr>

<h2>7. Bias–Variance Tradeoff</h2>
<p>
Hyperparameter tuning helps balance:
</p>
<ul>
  <li><b>Bias</b> – underfitting</li>
  <li><b>Variance</b> – overfitting</li>
</ul>

<hr>

<h2>8. Common Hyperparameters by Algorithm</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>Algorithm</th>
    <th>Important Hyperparameters</th>
  </tr>
  <tr>
    <td>Linear Regression</td>
    <td>Regularization (alpha)</td>
  </tr>
  <tr>
    <td>SVM</td>
    <td>C, kernel, gamma</td>
  </tr>
  <tr>
    <td>Decision Tree</td>
    <td>max_depth, min_samples_split</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>n_estimators, max_features</td>
  </tr>
  <tr>
    <td>Gradient Boosting</td>
    <td>learning_rate, n_estimators</td>
  </tr>
</table>

<hr>

<h2>9. Advantages of Hyperparameter Tuning</h2>
<ul>
  <li>Improves model performance</li>
  <li>Enhances generalization</li>
  <li>Provides optimal configurations</li>
</ul>

<hr>

<h2>10. Disadvantages of Hyperparameter Tuning</h2>
<ul>
  <li>Computationally expensive</li>
  <li>Time-consuming</li>
  <li>Requires expertise</li>
</ul>

<hr>

<h2>11. When to Use Hyperparameter Tuning</h2>
<ul>
  <li>Model selection</li>
  <li>Performance optimization</li>
  <li>Competitive ML tasks</li>
</ul>

<hr>

<h2>12. Summary</h2>
<p>
<b>Hyperparameter Tuning</b> systematically searches for the best hyperparameter
values to improve model accuracy, balance bias and variance, and ensure
strong generalization on unseen data.
</p>



<h1>⚖️ Data Balancing – Detailed Explanation</h1>

<h2>1. What is Data Balancing?</h2>
<p>
<b>Data Balancing</b> is the process of adjusting the class distribution in a dataset
so that all classes are represented fairly.
</p>
<p>
It is mainly used in <b>classification problems</b> where one class appears
significantly more frequently than others.
</p>

<hr>

<h2>2. What is Imbalanced Data?</h2>
<p>
A dataset is said to be <b>imbalanced</b> when one class dominates the dataset.
</p>

<p>
Example:
</p>
<ul>
  <li>90% → Class 0</li>
  <li>10% → Class 1</li>
</ul>

<p>
Such imbalance can cause the model to be biased toward the majority class.
</p>

<hr>

<h2>3. Why is Data Balancing Important?</h2>

<h3>Problems with Imbalanced Data</h3>
<ul>
  <li>High accuracy but poor real performance</li>
  <li>Minority class is ignored</li>
  <li>High false negatives</li>
</ul>

<h3>Benefits of Data Balancing</h3>
<ul>
  <li>Improves minority class prediction</li>
  <li>Reduces model bias</li>
  <li>Better recall and F1-score</li>
</ul>

<hr>

<h2>4. Types of Data Balancing Techniques</h2>

<h3>1. Undersampling</h3>
<p>
Undersampling reduces the number of samples in the majority class.
</p>

<p>
Advantages:
</p>
<ul>
  <li>Faster training</li>
</ul>

<p>
Disadvantages:
</p>
<ul>
  <li>Loss of useful information</li>
</ul>

<hr>

<h3>2. Oversampling</h3>
<p>
Oversampling increases the number of samples in the minority class.
</p>

<p>
Advantages:
</p>
<ul>
  <li>Preserves majority class data</li>
</ul>

<p>
Disadvantages:
</p>
<ul>
  <li>Risk of overfitting</li>
</ul>

<hr>

<h3>3. SMOTE (Synthetic Minority Over-sampling Technique)</h3>
<p>
<b>SMOTE</b> generates new synthetic samples for the minority class instead
of duplicating existing ones.
</p>

<p>
Advantages:
</p>
<ul>
  <li>Reduces overfitting</li>
  <li>Creates realistic samples</li>
</ul>

<p>
Disadvantages:
</p>
<ul>
  <li>Can generate noisy samples</li>
</ul>

<hr>

<h3>4. Class Weighting</h3>
<p>
Assigns higher weights to minority classes during model training.
</p>

<p>
This tells the model that misclassifying minority samples is more costly.
</p>

<hr>

<h2>5. Data Balancing vs No Balancing</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>Aspect</th>
    <th>No Balancing</th>
    <th>With Balancing</th>
  </tr>
  <tr>
    <td>Bias</td>
    <td>High</td>
    <td>Low</td>
  </tr>
  <tr>
    <td>Minority Class Recall</td>
    <td>Poor</td>
    <td>Improved</td>
  </tr>
  <tr>
    <td>Model Reliability</td>
    <td>Low</td>
    <td>High</td>
  </tr>
</table>

<hr>

<h2>6. When to Apply Data Balancing</h2>
<ul>
  <li>Before model training</li>
  <li>Only on training data (not test data)</li>
  <li>After train-test split</li>
</ul>

<hr>

<h2>7. Evaluation Metrics for Imbalanced Data</h2>
<ul>
  <li>Precision</li>
  <li>Recall</li>
  <li>F1-score</li>
  <li>ROC-AUC</li>
</ul>

<hr>

<h2>8. Common Data Balancing Techniques by Algorithm</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>Algorithm</th>
    <th>Recommended Balancing Method</th>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>Class Weighting</td>
  </tr>
  <tr>
    <td>Decision Tree</td>
    <td>SMOTE / Class Weight</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>Class Weight</td>
  </tr>
  <tr>
    <td>SVM</td>
    <td>Class Weight</td>
  </tr>
</table>

<hr>

<h2>9. Advantages of Data Balancing</h2>
<ul>
  <li>Improves minority class performance</li>
  <li>Reduces biased predictions</li>
  <li>Improves model fairness</li>
</ul>

<hr>

<h2>10. Disadvantages of Data Balancing</h2>
<ul>
  <li>Oversampling may cause overfitting</li>
  <li>Undersampling may lose information</li>
  <li>SMOTE may introduce noise</li>
</ul>

<hr>

<h2>11. When NOT to Use Data Balancing</h2>
<ul>
  <li>When classes are already balanced</li>
  <li>When imbalance reflects real-world importance</li>
</ul>

<hr>

<h2>12. Summary</h2>
<p>
<b>Data Balancing</b> ensures fair class representation in imbalanced datasets
by using techniques like undersampling, oversampling, SMOTE, and class weighting,
leading to better and more reliable model performance.
</p>


<h1>Clustering Algorithms: K-Means & Hierarchical</h1>

<hr>

<h2>🔹 1. K-Means Clustering</h2>

<h3>📌 What is K-Means?</h3>
<p>
K-Means is an <b>unsupervised learning algorithm</b> used to divide data into 
<b>K distinct clusters</b> based on similarity.
</p>
<p><b>Goal:</b> Group similar data points together and separate different ones.</p>

<h3>⚙️ How K-Means Works</h3>
<ol>
<li>Choose the number of clusters <b>K</b></li>
<li>Initialize <b>K centroids</b> randomly</li>
<li>Assign each data point to the nearest centroid</li>
<li>Recalculate centroids as the mean of assigned points</li>
<li>Repeat until centroids stop changing</li>
</ol>

<h3>📊 Example</h3>
<p>
Customer segmentation based on Age and Income:
<ul>
<li>Low income</li>
<li>Medium income</li>
<li>High income</li>
</ul>
</p>

<h3>📐 Objective Function</h3>
<p>Minimizes <b>WCSS (Within Cluster Sum of Squares)</b></p>

<h3>📈 Choosing K</h3>
<p><b>Elbow Method:</b> Select K where the graph bends like an elbow.</p>

<h3>✅ Advantages</h3>
<ul>
<li>Simple and fast</li>
<li>Works well for large datasets</li>
<li>Easy to implement</li>
</ul>

<h3>❌ Disadvantages</h3>
<ul>
<li>Need to choose K manually</li>
<li>Sensitive to initial centroids</li>
<li>Not good for non-spherical clusters or outliers</li>
</ul>

<h3>🧠 Use Cases</h3>
<ul>
<li>Customer segmentation</li>
<li>Image compression</li>
<li>Market analysis</li>
</ul>

<hr>

<h2>🔹 2. Hierarchical Clustering</h2>

<h3>📌 What is Hierarchical Clustering?</h3>
<p>
Builds a <b>tree-like structure (dendrogram)</b> of clusters.
No need to specify number of clusters initially.
</p>

<h3>🔁 Types</h3>
<ul>
<li><b>Agglomerative (Bottom-Up)</b></li>
<li><b>Divisive (Top-Down)</b></li>
</ul>

<h3>⚙️ Agglomerative Steps</h3>
<ol>
<li>Start with each point as its own cluster</li>
<li>Compute distances between clusters</li>
<li>Merge closest clusters</li>
<li>Repeat until one cluster remains</li>
</ol>

<h3>📏 Linkage Methods</h3>
<ul>
<li>Single Linkage (minimum distance)</li>
<li>Complete Linkage (maximum distance)</li>
<li>Average Linkage</li>
<li>Ward’s Method (minimizes variance)</li>
</ul>

<h3>🌳 Dendrogram</h3>
<p>
A tree diagram showing how clusters merge. You can cut the tree at a level 
to get desired clusters.
</p>

<h3>✅ Advantages</h3>
<ul>
<li>No need to choose K initially</li>
<li>Works well for small datasets</li>
<li>Provides hierarchy</li>
</ul>

<h3>❌ Disadvantages</h3>
<ul>
<li>Slow for large datasets</li>
<li>Not scalable</li>
<li>Cannot undo merging</li>
</ul>

<h3>🧠 Use Cases</h3>
<ul>
<li>Gene classification</li>
<li>Document clustering</li>
<li>Image segmentation</li>
</ul>

<hr>

<h2>🔥 K-Means vs Hierarchical</h2>

<table border="1" cellpadding="8" cellspacing="0">
<tr>
<th>Feature</th>
<th>K-Means</th>
<th>Hierarchical</th>
</tr>
<tr>
<td>Type</td>
<td>Partition-based</td>
<td>Tree-based</td>
</tr>
<tr>
<td>Need K?</td>
<td>Yes</td>
<td>No (initially)</td>
</tr>
<tr>
<td>Speed</td>
<td>Fast</td>
<td>Slow</td>
</tr>
<tr>
<td>Scalability</td>
<td>High</td>
<td>Low</td>
</tr>
<tr>
<td>Output</td>
<td>Flat clusters</td>
<td>Dendrogram</td>
</tr>
<tr>
<td>Flexibility</td>
<td>Low</td>
<td>High</td>
</tr>
</table>

<hr>

<h2>💡 When to Use</h2>

<p><b>Use K-Means:</b></p>
<ul>
<li>Large datasets</li>
<li>K is known</li>
<li>Need fast results</li>
</ul>

<p><b>Use Hierarchical:</b></p>
<ul>
<li>Small datasets</li>
<li>Need hierarchy</li>
<li>K is unknown</li>
</ul>

<hr>

<h2>🎯 Intuition</h2>
<ul>
<li><b>K-Means:</b> Group into K clusters</li>
<li><b>Hierarchical:</b> Build a cluster tree</li>
</ul>

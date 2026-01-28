
<h2>1. Machine Learning?</h2>
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

<h2>7. Loss Function</h2>
<p>
A loss function measures how far the predicted values are from the actual values.
</p>

<h3>Mean Squared Error (MSE)</h3>
<p>
MSE is commonly used in regression problems.
</p>

<pre>
MSE = (1/n) Σ (actual − predicted)²
</pre>

<p>
Lower loss indicates a better model.
</p>

<hr>

<h2>8. Gradient Descent</h2>
<p>
Gradient Descent is an optimization algorithm used to minimize the loss function.
</p>

<p>
It iteratively updates model parameters (m and c) by moving in the direction of
the steepest decrease in loss.
</p>

<h3>Update Rules</h3>
<pre>
m = m − α * ∂(Loss)/∂m
c = c − α * ∂(Loss)/∂c
</pre>

<hr>

<h2>9. Learning Rate</h2>
<p>
The learning rate (α) controls how big a step gradient descent takes.
</p>

<ul>
  <li>Too large → may overshoot the minimum</li>
  <li>Too small → slow convergence</li>
  <li>Optimal value → fast and stable learning</li>
</ul>

<p>
Common values include <b>0.01</b>, <b>0.001</b>, and <b>0.1</b>.
</p>

<hr>

<h2>10. R² Score (Coefficient of Determination)</h2>
<p>
R² score measures how well the regression model explains the variance in the data.
</p>

<h3>Interpretation</h3>
<ul>
  <li>R² = 1 → Perfect prediction</li>
  <li>R² ≈ 0.9 → Very good</li>
  <li>R² ≈ 0 → Poor model</li>
  <li>R² &lt; 0 → Worse than random</li>
</ul>

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



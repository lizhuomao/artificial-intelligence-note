# 什么是机器学习

​	task, performance measure,  experience;

+ Machine Learning algorithms:

  + Supervised learning 

  + Unsupervised learning 

+ Others: Reinforcememt learing, recommender Systems

## Supervised learning 监督学习

​	Given the "right answer" for each example in the data.

​	regression problem (回归问题)

​	**Regression**:Predict continuous valued output

​	**Classification**:Discrete valued output(0 or 1)

### Linear regression

​	m = Number of training examples

​	x = "input" variable/features

​	y = "output" variable/"target"variable

+ supervised learning work struction:

  ![image-20220310232702177](assets/image-20220310232702177.png)

  another name: univariate linear regeression  $h_θ(x)=θ_0 +θ_1x$

+  hyppothsis
  
  Goal:
  $$
  {minimize\atop θ_0, θ_1}\frac{1}{2m}\sum_{i= 1}^m(h_θ(x^{(i)}) - y^{(i)})^2
$$
  *tips:平方数学性质好，结果有正有负，平方导数处处存在*

  预测值和实际值得平方误差和

  *tips:这里的1/2m是为了开导 时刚好抵消平方，对优化结果来说无影响*

  线性回归的整体目标函数

  cost function (squared error function) 代价函数 :$\frac{1}{2m}\sum_{i= 1}^m(h_θ(x^{(i)}) - y^{(i)})^2$
  
  

## Unsupervised learning 无监督学习

​	clustering 聚类...


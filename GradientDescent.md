# Notion of GD

##### GD

- Iteration and approximate Algorithm
- Using in Un-constraint and convex function

##### target

- Using solution the cost Function parameters of $\theta$
- $\theta$ can be random.choice,or initial is 0
- $\theta=\theta-\alpha\frac{\partial{J(\theta)}}{\theta}$
  - Note:$\alpha$  is learning rate，max($\alpha$) can not get a answer, min($\alpha$) the solution procession slower
  - elaryStoping conditon
    - Loss is equal last value
    - iteration number
    - seting a threshold
  - Initial lr:
    - according to the $\theta$ changing control the lr
    - Dynamic learning rate:Such as during the first :the lr can be set a max value of inf,when $\theta$ changing at descent we can reduce a half of lr
    - Initial zeros 
    - experience,crossValidation,1e-10

##### common GD

- SGD
  - Onece time just update a sample of parameters
  - Can jump out the local opitimal
  - May increase the error of a sample
  - The procession of update is unsoomthing
  - The path is different
  - **formula**:$\frac{\partial{J(\theta)}}{\partial(\theta_{i})}=(h_{\theta}(x)-y)X_{j}$
  -  $for each i = 1{\to}m\\{\theta_{j}=\theta_{j}+\alpha(y^i-h_{\theta}(x^i))x^j}$
- BGD
  - All gradient :sum(subset gradient)
  - Once time need to upadte parameters of all datas
  - ![image](/Users/afei/PycharmProjects/deepBlueAILecture/python基础/NLP/image/BGD.png)
- MBGD
  - Onece time just get epochs/iteration_num,b just 10 sample,step+=10
  - ![image](/Users/afei/PycharmProjects/deepBlueAILecture/python基础/NLP/image/MBGD.png)



##### compare wit each other

-  SGD faster  than BGD,just iteration less step
- SGD may have more nums local optimal,but it can be jump out the local optimal
- SGD maybe shock when convergence
- BGD must get a global optimal at learning Regression model
- SGD is smoothing, the error unchanging
- MBGD and SGD is shocking
- ![image](/Users/afei/PycharmProjects/deepBlueAILecture/python基础/NLP/image/GDcompare.png)



##### GD Just a Method

- ![iamge](/Users/afei/PycharmProjects/deepBlueAILecture/python基础/NLP/image/a methold.png)
- <font color='red'> **GD just a methods for solving parameters,not a model** </font>


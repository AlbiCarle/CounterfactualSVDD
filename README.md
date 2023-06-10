# CounterfactualSVDD

Increasingly in recent times, the mere prediction of a machine learning algorithm is considered insufficient to gain complete control over the event being predicted. A machine learning algorithm should be considered reliable in the way it allows to extract more knowledge and information than just having a prediction at hand. In this perspective, the counterfactual theory plays a central role. By definition, a counterfactual is the smallest variation of the input such that it changes the predicted behaviour.
In this work, we addressed counterfactuals through Support Vector Data Description (SVDD) empowered by explainability and metric for assessing the counterfactual quality. 
As a matter of fact, to generate counterfactual explanations using SVDD, an optimization problem involving kernels, design parameters and evaluation metrics must be solved. In this sense, SVDD proved to be a good approach for generating counterfactual eXpalanations since it is possible to prove that analytical and exact solution may be found (under relaxed hypothesis, but suggestions can lead to think that a generalization can be found). 
The results obtained are more than encouraging and suggest that the proposed method can compete with those in the state of the art.

In this repository you will find MATLAB and Python code for implementing counterfactual analysis using SVDD. Specifically

-) Utils contains the main matlab functions to run the code:

    -) TC_SVDD_TRAINING_NEW.m contains the SVDD training algorithm;
    
    -) TC_SVDD_TEST.m contains the SVDD test algorithm;
    
    -) KernelMatrix.m contains the script to use kernels functions;
    
    -) TestObject_N,m contains the code to compute the distance from the center of the SVDD hyperspheres;
    
    -) FNR_TCSVDD.m contains the code to control the number of false negatives in the classification;
    
    -) holdoutCVKernTCSVDD.m contains the code to optimize the SVDD hyperparameters
    
    -) SquareDist.m is a tool function.
    
    
-) eXamples contains 

    -) an example for extracting counterfactuals from Vehicle Platooning (example.m);
    
    -) an healthcare dataset example (Diabetes_example.m);
    
    -) a folder containing the application of CounterfactualSVDD to an smart mobility scenario (MTT).
    
    -) A brief presentation of the method is reported in Counterfactual_eXplanations.pdf
 


Please refer to 

-) A. Carlevaro, M. Lenatti, A. Paglialonga and M. Mongelli, "Counterfactual Building and Evaluation via eXplainable Support Vector Data Description," in IEEE Access, vol. 10, pp. 60849-60861, 2022, doi: 10.1109/ACCESS.2022.3180026. 

-) M. Lenatti, A. Carlevaro, A. Guergachi, K. Keshavjee, M. Mongelli, A. Paglialonga, "Individualized minimum viable recommendations for Type 2
Diabetes prevention using counterfactual explanations", submitted to PLOS ONE.






![example11](https://user-images.githubusercontent.com/99175531/173586503-5dec263d-a2bf-4806-a0d1-596599b5933a.png)



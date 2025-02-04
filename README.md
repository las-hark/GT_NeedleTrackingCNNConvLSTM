# GT_NeedleTrackingCNNConvLSTM
see code in https://github.com/las-hark/GT_NeedleTrackingCNNConvLSTM/tree/master/tes

![image](https://github.com/user-attachments/assets/2d0179bd-5987-490e-865a-caadfff2aad5)

In ultrasound-guided surgery, the interventional needle needs to reach the target site safely, effectively, and accurately, for which the operating physician requires extensive training and experience, while the limited quality of ultrasound imaging is prone to noise, artifacts, and other interference, increasing the probability of operational errors or biased judgments. 

In this experiment, 2D B-mode ultrasound-guided puncture is performed on isolated muscle tissue as an experimental object. The collected ultrasound video is acquired and further pre-processed to create a dataset. 

![image](https://github.com/user-attachments/assets/56e66acf-18ff-453a-b465-52d1ed80c28f)
![image](https://github.com/user-attachments/assets/d4b34fc4-41f0-4e21-8837-7d093cdcd369)

Based on the dataset, I propose a traditional image-processing algorithm and a deep-learning algorithm to localize the needle.

The traditional algorithm is based on morphological feature analysis, showing the result as below.
![image](https://github.com/user-attachments/assets/49d47946-8e0a-4811-8f0d-8b939139c91c)

For better performance, I design a CNN-ConvLSTM model based on the tip-enhanced sequence, apply ReLU activation function and depthwise separable convolution to lighten it, and compare the algorithm performance under different hyperparameter settings to obtain a lightweight CNN-ConvLSTM model.

![image](https://github.com/user-attachments/assets/662bfc38-1553-4bcc-a17b-bd7de733895c)
![image](https://github.com/user-attachments/assets/2d0179bd-5987-490e-865a-caadfff2aad5)

For the interventional ultrasound dataset collected by myself, plz reach out

# Application example: Photo OCR

## Problem description and pipeline

如何让计算机读取处图片中的文字

![image-20220503212542645](assets/image-20220503212542645.png)

拿到一个机器学习任务后要对任务进行分解

![image-20220503215013084](assets/image-20220503215013084.png)

将任务分解为多个模块，分工解决

![image-20220503215258512](assets/image-20220503215258512.png)

## Sliding windows

通过一个窗口在图片中滑动将，每一个都进行检测

### Text detection

![image-20220503233451037](assets/image-20220503233451037.png)

![image-20220503233533037](assets/image-20220503233533037.png)

单个框相距够近就会合并

### 1D Sliding window for character segmentation

![image-20220503233657956](assets/image-20220503233657956.png)

![image-20220503233805021](assets/image-20220503233805021.png)

## Getting lots of data: Artificial data synthesis

### Character recognition

![image-20220503233915861](assets/image-20220503233915861.png)

### Artificial data synthesis for photo OCR

通过变换图片得到新的数据集

![image-20220503234054973](assets/image-20220503234054973.png)

Synthesizing data by introducing distortions

![image-20220503234305225](assets/image-20220503234305225.png)

![image-20220503234431493](assets/image-20220503234431493.png)

要选择合适的变换

### Discussion on getting more data

1. Make sure you have a low bias classifier before expending the effort. (Plot learning curves). E.g. keep increasing the number of features/number of hidden units in neural network until you have a low bias classifier
2. "How much work would it be to get 10x as much data as we currently have?"
   - Artificial data synthesis
   - Collect/label it yourself
   - "Crowd source"

## Ceiling analysis: What part of the pipeline to work on next

把时间画在刀刃上，更值得

![image-20220503235543251](assets/image-20220503235543251.png)

![image-20220503235558644](assets/image-20220503235558644.png)
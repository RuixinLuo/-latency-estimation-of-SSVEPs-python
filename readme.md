> # Latency Estimation of SSVEP -python

#### SSVEP latency estimation based on paper [1]_.

#### Adapted from https://github.com/iqgnat/SSVEP_phase_latency from MATLAB to python

> [1] Jie P , Gao X , Fang D , et al. Enhancing the classification accuracy of steady-state visual evoked potential-based brain-computer interfaces using phase constrained canonical correlation analysis[J].Journal of Neural Engineering, 2011, 8(3):036027.

## Dataset

#### Benchmark dataset [2]_ from Tsinghua university.

>  [2] Wang Y , Chen X , Gao X , et al. A Benchmark Dataset for SSVEP-Based Brain-Computer Interfaces[J].IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2017, 25(10):1746-1752.

## results

#### if everything is ok, you will get the mean_latency ≈ 136ms

#### the result is same as the source code from https://github.com/iqgnat/SSVEP_phase_latency

##### ***2022.6.23  we delete -180 when solving for the phase, which is different from the source code.

## Acknowledgement

#### Thanks to [TangQi]([iqgnat (Tang Qi) (github.com)](https://github.com/iqgnat)), the author of the source code, for his patience in responding to my questions.

## email

#### ruixin_luo@tju.edu.cn




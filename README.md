# AI_TSP
Using Christofides heuristic algorithm to find approximate solution for TSP(Travelling Salesman problem)
<img width="1280" alt="Route" src="https://github.com/NEU20205988John/AI_TSP/assets/80146486/f57dc550-1b3a-4f76-a0a1-1c30f4b1ae04">

## Abstract

本文针对TSP问题，采用Christofides启发式算法，在多项式时间内求出代价小于等于最优解代价1.5倍之内的一个近似解。并在近似解的基础上采用了2-OPT优化方法和自行研究的优化算法。最终得到了较为优秀的结果，和精确解十分接近。报告中主要包括TSP相关背景介绍、编程环境、技术路线、项目运行结果以及项目心得。

项目运行得到的结果与精确解十分近似，并且程序可以在多项式时间内得到确定的、准确的结果。在使用C++编程语言实现克里斯托菲德算法的同时，可以促进图论相关知识的复习。最后采用python编程语言进行了TSP路径的绘制。项目综合可行性高、算法时间复杂度低。在MacBook Pro上运行项目，可以在15ms内得到代价为155.153的最终结果。（单位以经纬度下的欧氏距离之和计算）

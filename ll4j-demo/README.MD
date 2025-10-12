示例
===

原本的 LL4j 模型在 `ll4j-huzpsb/src/main/resource` 下

如果要从头炼一个，可以解压 fashionMnist.zip，然后运行代码里各种 main 函数看看。

包 `com.shinonometn.ml.ll4j.demo`
---

Classify 是测试模型的准确度；Train 是训练，会对 60k 个样本进行 128 轮测试。
在 M2 Macbook 上大概就是 14 秒一轮，不会很长。

包 `huzpsb.ll4j.samples`
---

这些是原版程序的样例
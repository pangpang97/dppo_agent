dppo_cat_mountainCar.py 在gym的mountainCar-V0上的测试
dppo_cat_realGame.py 在我们的环境上的代码，较之上面的代码主要的改动为1）在choose_action方法加了action mask；2）网络的unit不一样。
environment_all.py 这个是我们的游戏与真实的游戏之间的转化的脚本
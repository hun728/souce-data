%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = xlsread('数据集.xlsx');
ses = xlsread('预测集.xlsx');
%%  划分训练集和测试集
temp = 1:1:181;

P_train = res(temp(1: 403), 1: 5)';
T_train = res(temp(1: 403), 6)';
M = size(P_train, 2);

P_test = res(temp(404: end), 1: 5)';
T_test = res(temp(404: end), 6)';
N = size(P_test, 2);
P_prediction = ses(1: end, 1: 5)';
T_prediction = ses(1: end, 6)';
L = size(P_prediction, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
p_prediction = mapminmax('apply', P_prediction, ps_input);
[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);
t_prediction = mapminmax('apply', T_prediction, ps_output);


%%  节点个数
inputnum  = size(p_train, 1);  % 输入层节点数
hiddennum = 11;                 % 隐藏层节点数
outputnum = size(t_train, 1);  % 输出层节点数

%%  建立网络
net = newff(p_train, t_train, hiddennum);

%%  设置训练参数
net.trainParam.epochs     = 1000;      % 训练次数
net.trainParam.goal       = 1e-6;      % 目标误差
net.trainParam.lr         = 0.01;      % 学习率
net.trainParam.showWindow = 0;         % 关闭窗口

%%  参数初始化
c1      = 2;       % 学习因子
c2      = 2;       % 学习因子
maxgen  =   50;        % 种群更新次数  
sizepop =   20;        % 种群规模
Vmax    =  1.0;        % 最大速度
Vmin    = -1.0;        % 最小速度
popmax  =  1.0;        % 最大边界
popmin  = -1.0;        % 最小边界

%%  节点总数
numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;

for i = 1 : sizepop
    pop(i, :) = rands(1, numsum);  % 初始化种群
    V(i, :) = rands(1, numsum);    % 初始化速度
    fitness(i) = fun(pop(i, :), hiddennum, net, p_train, t_train);
    fitness1(i) = fun1(pop(i, :), hiddennum, net, p_train, t_train);
end

%%  个体极值和群体极值
[fitnesszbest, bestindex] = min(fitness);
zbest = pop(bestindex, :);     % 全局最佳
gbest = pop;                   % 个体最佳
fitnessgbest = fitness;   % 个体最佳适应度值
fitnessgbest1 = fitness1; 
BestFit = fitnesszbest;        % 全局最佳适应度值

%%  迭代寻优
for i = 1: maxgen
    for j = 1: sizepop
        
        % 速度更新
        V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        V(j, (V(j, :) > Vmax)) = Vmax;
        V(j, (V(j, :) < Vmin)) = Vmin;
        
        % 种群更新
        pop(j, :) = pop(j, :) + 0.2 * V(j, :);
        pop(j, (pop(j, :) > popmax)) = popmax;
        pop(j, (pop(j, :) < popmin)) = popmin;
        
        % 自适应变异
        pos = unidrnd(numsum);
        if rand > 0.85
            pop(j, pos) = rands(1, 1);
        end
        
        % 适应度值
        fitness(j) = fun(pop(j, :), hiddennum, net, p_train, t_train);
        fitness1(j) = fun1(pop(j, :), hiddennum, net, p_train, t_train);

    end
    
    for j = 1 : sizepop

        % 个体最优更新
        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);
            fitnessgbest(j) = fitness(j);
            fitnessgbest1(j) = fitness1(j);

        end

        % 群体最优更新 
        if fitness(j) < fitnesszbest
            zbest = pop(j, :);
            fitnesszbest = fitness(j);
            fitnesszbest1 = fitness1(j);
        end

    end

    BestFit = [BestFit, fitnesszbest];    
end

fitnessgbest11 = fitnessgbest1';
gbest1 = [gbest,fitnessgbest11];

%%  设置优化参数
S1 = 11;           %  隐藏层节点个数 
gen = 50;                       % 遗传代数
pop_num = 20;                    % 种群规模
S = size(p_train, 1) * S1 + S1 * size(t_train, 1) + S1 + size(t_train, 1);
                                % 优化参数个数
bounds = ones(S, 1) * [-1, 1];  % 优化变量边界

%%  初始化种群
prec = [1e-6, 1];               % epslin 为1e-6, 实数编码
normGeomSelect = 0.09;          % 选择函数的参数
arithXover = 2;                 % 交叉函数的参数
nonUnifMutation = [2 gen 3];    % 变异函数的参数
initPpp = gbest1;
 



%%  优化算法
[Bestpop, endPop, bPop, trace] = ga(bounds, 'gabpEval', [], initPpp, [prec, 0], 'maxGenTerm', gen,...
                           'normGeomSelect', normGeomSelect, 'arithXover', arithXover, ...
                           'nonUnifMutation', nonUnifMutation);

%%  获取最优参数
[val, W1, B1, W2, B2] = gadecod(Bestpop);

%%  参数赋值
net.IW{1, 1} = W1;
net.LW{2, 1} = W2;
net.b{1}     = B1;
net.b{2}     = B2;

%%  模型训练
net.trainParam.showWindow = 1;       % 打开训练窗口
net = train(net, p_train, t_train);  % 训练模型





%%  仿真测试
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );
t_sim3 = sim(net, p_prediction );
hb = [t_sim1,t_sim2,t_sim3];

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_sim3 = mapminmax('reverse', t_sim3, ps_output);
HB = mapminmax('reverse', hb, ps_output);

%%  均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

%%  优化迭代曲线
figure
plot(trace(:, 1), 1 ./ trace(:, 2), 'LineWidth', 1.5);
xlabel('迭代次数');
ylabel('适应度值');
string = {'适应度变化曲线'};
title(string)
grid on

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比';['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  相关指标计算
%  R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])

%  MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])
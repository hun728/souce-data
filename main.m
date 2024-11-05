%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��������
res = xlsread('���ݼ�.xlsx');
ses = xlsread('Ԥ�⼯.xlsx');
%%  ����ѵ�����Ͳ��Լ�
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

%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
p_prediction = mapminmax('apply', P_prediction, ps_input);
[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);
t_prediction = mapminmax('apply', T_prediction, ps_output);


%%  �ڵ����
inputnum  = size(p_train, 1);  % �����ڵ���
hiddennum = 11;                 % ���ز�ڵ���
outputnum = size(t_train, 1);  % �����ڵ���

%%  ��������
net = newff(p_train, t_train, hiddennum);

%%  ����ѵ������
net.trainParam.epochs     = 1000;      % ѵ������
net.trainParam.goal       = 1e-6;      % Ŀ�����
net.trainParam.lr         = 0.01;      % ѧϰ��
net.trainParam.showWindow = 0;         % �رմ���

%%  ������ʼ��
c1      = 2;       % ѧϰ����
c2      = 2;       % ѧϰ����
maxgen  =   50;        % ��Ⱥ���´���  
sizepop =   20;        % ��Ⱥ��ģ
Vmax    =  1.0;        % ����ٶ�
Vmin    = -1.0;        % ��С�ٶ�
popmax  =  1.0;        % ���߽�
popmin  = -1.0;        % ��С�߽�

%%  �ڵ�����
numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;

for i = 1 : sizepop
    pop(i, :) = rands(1, numsum);  % ��ʼ����Ⱥ
    V(i, :) = rands(1, numsum);    % ��ʼ���ٶ�
    fitness(i) = fun(pop(i, :), hiddennum, net, p_train, t_train);
    fitness1(i) = fun1(pop(i, :), hiddennum, net, p_train, t_train);
end

%%  ���弫ֵ��Ⱥ�弫ֵ
[fitnesszbest, bestindex] = min(fitness);
zbest = pop(bestindex, :);     % ȫ�����
gbest = pop;                   % �������
fitnessgbest = fitness;   % ���������Ӧ��ֵ
fitnessgbest1 = fitness1; 
BestFit = fitnesszbest;        % ȫ�������Ӧ��ֵ

%%  ����Ѱ��
for i = 1: maxgen
    for j = 1: sizepop
        
        % �ٶȸ���
        V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        V(j, (V(j, :) > Vmax)) = Vmax;
        V(j, (V(j, :) < Vmin)) = Vmin;
        
        % ��Ⱥ����
        pop(j, :) = pop(j, :) + 0.2 * V(j, :);
        pop(j, (pop(j, :) > popmax)) = popmax;
        pop(j, (pop(j, :) < popmin)) = popmin;
        
        % ����Ӧ����
        pos = unidrnd(numsum);
        if rand > 0.85
            pop(j, pos) = rands(1, 1);
        end
        
        % ��Ӧ��ֵ
        fitness(j) = fun(pop(j, :), hiddennum, net, p_train, t_train);
        fitness1(j) = fun1(pop(j, :), hiddennum, net, p_train, t_train);

    end
    
    for j = 1 : sizepop

        % �������Ÿ���
        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);
            fitnessgbest(j) = fitness(j);
            fitnessgbest1(j) = fitness1(j);

        end

        % Ⱥ�����Ÿ��� 
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

%%  �����Ż�����
S1 = 11;           %  ���ز�ڵ���� 
gen = 50;                       % �Ŵ�����
pop_num = 20;                    % ��Ⱥ��ģ
S = size(p_train, 1) * S1 + S1 * size(t_train, 1) + S1 + size(t_train, 1);
                                % �Ż���������
bounds = ones(S, 1) * [-1, 1];  % �Ż������߽�

%%  ��ʼ����Ⱥ
prec = [1e-6, 1];               % epslin Ϊ1e-6, ʵ������
normGeomSelect = 0.09;          % ѡ�����Ĳ���
arithXover = 2;                 % ���溯���Ĳ���
nonUnifMutation = [2 gen 3];    % ���캯���Ĳ���
initPpp = gbest1;
 



%%  �Ż��㷨
[Bestpop, endPop, bPop, trace] = ga(bounds, 'gabpEval', [], initPpp, [prec, 0], 'maxGenTerm', gen,...
                           'normGeomSelect', normGeomSelect, 'arithXover', arithXover, ...
                           'nonUnifMutation', nonUnifMutation);

%%  ��ȡ���Ų���
[val, W1, B1, W2, B2] = gadecod(Bestpop);

%%  ������ֵ
net.IW{1, 1} = W1;
net.LW{2, 1} = W2;
net.b{1}     = B1;
net.b{2}     = B2;

%%  ģ��ѵ��
net.trainParam.showWindow = 1;       % ��ѵ������
net = train(net, p_train, t_train);  % ѵ��ģ��





%%  �������
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );
t_sim3 = sim(net, p_prediction );
hb = [t_sim1,t_sim2,t_sim3];

%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_sim3 = mapminmax('reverse', t_sim3, ps_output);
HB = mapminmax('reverse', hb, ps_output);

%%  ���������
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

%%  �Ż���������
figure
plot(trace(:, 1), 1 ./ trace(:, 2), 'LineWidth', 1.5);
xlabel('��������');
ylabel('��Ӧ��ֵ');
string = {'��Ӧ�ȱ仯����'};
title(string)
grid on

%%  ��ͼ
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('��ʵֵ','Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('��ʵֵ','Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�';['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  ���ָ�����
%  R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;

disp(['ѵ�������ݵ�R2Ϊ��', num2str(R1)])
disp(['���Լ����ݵ�R2Ϊ��', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;

disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1)])
disp(['���Լ����ݵ�MAEΪ��', num2str(mae2)])

%  MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;

disp(['ѵ�������ݵ�MBEΪ��', num2str(mbe1)])
disp(['���Լ����ݵ�MBEΪ��', num2str(mbe2)])
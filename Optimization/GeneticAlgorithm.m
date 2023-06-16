clear;
clc;

data = readmatrix("C:\Users\fatim\Box\Habbab Fatima\Experiment set 6\Regression\OOS\SVR\test60.xlsx");
ret = tick2ret(data);

% Set transaction cost
tCost = 0.002;

% Portfolio parameters
N = size(ret,2); % number of assets
M = size(ret,1); % number of observations
[R1,C1,Obs] = ewstats(ret); % expected return R
                            % expected covariance C
                            % number of observations Obs
volatility = std(ret);

actual_data = readmatrix("C:\Users\fatim\Box\Habbab Fatima\Experiment set 6\Regression\OOS\SVR\test60.xlsx");
actual_ret = tick2ret(actual_data);
[R2,C2,Obs2] = ewstats(actual_ret);

% Set risk-free rate
riskFree = 0.000019;
        
% GA parameters
numRuns = 100;
tournSize = 3;
popSize = 500;
genes = N;
mutRate = 0.1;
numGen = 25;
pop = (ones(genes, popSize) / genes)'; % equal weights
bestFitness = zeros(numGen, 1); % keeping track of the best fitness per generation
avgFitness = zeros(numGen, 1); % average population fitness per generation
bestWeights = zeros(numGen+1, genes);
bestWeights(1, :) = popSize(1,:);
testFitness = zeros(numRuns,1);
        
for n = 1:numRuns
for i = 1:numGen

    % calculate variance per GA individual
    varPerIndividual = zeros(popSize,1);
    returnPerIndividual = zeros(popSize,1);
    sharpeRatioPerIndividual = zeros(popSize,1);

    for k = 1:popSize
        returnPerIndividual(k) = pop(k,:) * R1';
        varPerIndividual(k) = pop(k,:) * C1 * pop(k,:)';
        sharpeRatioPerIndividual(k) = returnPerIndividual(k)./ sqrt(varPerIndividual(k));
    end

    fitness = sharpeRatioPerIndividual;

    avgFitness(i) = mean(fitness);
    bestFitness(i) = max(fitness);
    disp(['Generation: ', num2str(i), ', Best so far: ', num2str(max(fitness))]);

    % obtain tournament participants
    tournamentP1 = randi([1, popSize], popSize, tournSize);
    tournamentP2 = randi([1, popSize], popSize, tournSize);

    % tournament selection for parent 1
    [val, idx] = max(fitness(tournamentP1),[],2);
    p1 = zeros(popSize, genes);
    for k = 1:length(tournamentP1)
        p1(k,:) = pop(tournamentP1(k,idx(k)),:);
    end

    % tournament selection for parent 2
    [val, idx] = max(fitness(tournamentP2),[],2);
    p2 = zeros(popSize, genes);
    for k = 1:length(tournamentP2)
        p2(k,:) = pop(tournamentP2(k,idx(k)),:);
    end

    % crossover
    cutoff = randi([1, genes-1], popSize, 1); % cut-off point
    pop_temp = [p1(:,1:cutoff) p2(:, cutoff+1:end)];

    % mutation
    rowToMutate = randi([1, popSize], ceil(mutRate * popSize), 1);
    colToMutate = randi([1, genes], 1, ceil(mutRate * popSize));
    mutatedValue = rand(ceil(mutRate * popSize), 1);
    sz = size(pop);
    pop_temp(sub2ind(sz,rowToMutate,colToMutate')) = mutatedValue;

    % elitism
    [elitist, index] = max(fitness);
    pop_temp(1,:) = pop(index, :);
    bestWeights(i+1,:) = pop(index, :);

    % normalise weights
    pop_temp = pop_temp./sum(pop_temp,2);

    % copy intermediate population
    pop = pop_temp;
    pop_temp = [];
end

% Compute portfolio metrics
[portRisk(n,:),portReturn(n,:)] = portstats(R2,C2,bestWeights(end,:));
end

portReturn = portReturn .* 1.05;
sharpeRatio = (portReturn - riskFree)./ sqrt(portRisk);
results = [portReturn portRisk sharpeRatio];
disp(['Sharpe ratio: ', num2str(mean(sharpeRatio))])

%file = "C:\Users\fh20175\Box\Habbab Fatima\Experiment set 6\Optimization\OOS\KNNport.xlsx";
%writematrix(results,file);

[h p k] = kstest2(x1,x2);
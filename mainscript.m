% mainscript is rather short this time
pkg load statistics


% primary component count
comp_count = 80; 

[tvec tlab tstv tstl] = readSets(); 

% let's look at the first digit in the training set
imshow(1-reshape(tvec(1,:), 28, 28)');

% let's check labels in both sets
[unique(tlab)'; unique(tstl)']

% compute and perform PCA transformation
[mu trmx] = prepTransform(tvec, comp_count);
tvec = pcaTransform(tvec, mu, trmx);
tstv = pcaTransform(tstv, mu, trmx);

% let's shift labels by one to use labels directly as indices
tlab += 1;
tstl += 1;

% To successfully prepare ensemble you have to implement perceptron function
% I would use 10 first zeros and 10 fisrt ones 
% and only 2 first primary components
% It'll allow printing of intermediate results in perceptron function

%
% YOUR CODE GOES HERE - testing of the perceptron function
pzeros = tvec(tlab==1, 1:5);
nones = tvec(tlab==2, 1:5);
sp = perceptron(pzeros, nones);
%[-ones(10,1) -nones] * sp' < 0;


% training of the whole ensemble
[ovo, ovoerr] = trainOVOensamble(tvec, tlab, @perceptron);

% check your ensemble on train set
clab = unamvoting(tvec, ovo);
cfmx = confMx(tlab, clab)
compErrors(cfmx)

% repeat on test set
clab = unamvoting(tstv, ovo);
cfmx = confMx(tstl, clab)
compErrors(cfmx)


% YOUR CODE GOES HERE
sortedErrors = sortrows(ovoerr, 5);
worstPair = sortedErrors(end,1:2)
worstOvoRow = find(ovo(:, 1) == worstPair(1) & ovo(:, 2) == worstPair(2));

% cluster positive
pos = tvec(tlab==worstPair(1),:);
[idxPos, _, _, _] = kmeans (pos, 3);

% cluster negative
neg = tvec(tlab==worstPair(2),:);
[idxNeg, _, _, _] = kmeans (neg, 3);

[ovoWorst, ovoWorstErr] = trainClusters(worstPair, idxPos, idxNeg, pos, neg);


ovoNoWorst = ovo;

% remove worst classifier in ovo
ovoNoWorst(worstOvoRow,:) = [];

% learning set
clab = unamvoting2(tvec, worstPair, ovoNoWorst, ovoWorst);
cfmx = confMx(tlab, clab)
compErrors(cfmx)

% test set
clab = unamvoting2(tstv, worstPair, ovoNoWorst, ovoWorst);
cfmx = confMx(tstl, clab)
compErrors(cfmx)

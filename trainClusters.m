function [ovosp, ovoerr] = trainClusters(clsIds, idxPos, idxNeg, pos, neg)
  ovosp = zeros(9, 2 + 1 + columns(pos));
  pairs = [1 1; 1 2; 1 3; 2 1; 2 2; 2 3; 3 1; 3 2; 3 3];
  ovoerr = zeros(rows(pairs), 2 + 3);

  for i=1:rows(pairs)
    ovosp(i, 1:2) = clsIds;
    ovoerr(i, 1:2) = clsIds;
    
    posSamples = pos(idxPos==pairs(i,1), :);
    negSamples = neg(idxNeg==pairs(i,2), :);
    
	  [sp fp fn] = trainSelect(posSamples, negSamples, 5, @perceptron);
    
    errorRate = (fp + fn)/(rows(posSamples) + rows(negSamples));
	  ovoerr(i, 3:end) = [fp fn errorRate];
    
    ovosp(i, 3:end) = sp;
  end

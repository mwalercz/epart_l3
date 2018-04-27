function clab = unamvoting2 (tset, worstPair, ovoNoWorst, ovoWorst)
  % voting without worst classifier
  votesNoWorst = voting(tset, ovoNoWorst);

  % voting for worst classifier
  votesWorst = voting(tset, ovoWorst);
  [_ clabWorst] = max(votesWorst, [], 2);

  posIdx = clabWorst == 1;
  negIdx = clabWorst == 2;

  % add one for class that has most votes in worst classifier
  votesNoWorst(posIdx, worstPair(1)) += 1;
  votesNoWorst(negIdx, worstPair(2)) += 1;

  votesAll = votesNoWorst;

  % unamvoting
  labels = unique(ovoNoWorst(:, [1 2]));
  maxvotes = rows(labels) - 1; % unanimity voting in one vs. one scheme
  reject = max(labels) + 1;
  [maxVal clab] = max(votesAll, [], 2);
  clab(maxVal ~= maxvotes) = reject;
endfunction

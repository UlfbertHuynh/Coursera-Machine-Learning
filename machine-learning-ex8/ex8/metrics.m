function [precision recall f1] = metrics(yval, prediction)
true_pos = sum(yval == prediction & yval == 1);
false_pos = sum(yval != prediction & yval == 0);
false_neg = sum(yval != prediction & yval == 1);
precision = true_pos / (true_pos + false_pos);
recall = true_pos / (true_pos + false_neg);
f1 = 2 * ((precision * recall) / (precision + recall));
end
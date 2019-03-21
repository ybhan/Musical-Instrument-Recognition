%Written by Zhang Wenyu
function y = extract(x)
if (size(x,2)<200)
    y = [x,zeros(size(x,1),200-size(x,2))];
else
    y = x(:,1:200);
end
end


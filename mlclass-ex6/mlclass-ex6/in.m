function T = in(item,vectr)
for i = 1:length(vectr)
  if item == vectr(i)
    T = 1;
  else
    T = 0;
  end
end

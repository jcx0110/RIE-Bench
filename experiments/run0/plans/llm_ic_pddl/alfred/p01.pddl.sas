begin_version
3
end_version
begin_metric
0
end_metric
3
begin_variable
var0
-1
2
Atom handempty(left)
NegatedAtom handempty(left)
end_variable
begin_variable
var1
-1
2
Atom handempty(right)
NegatedAtom handempty(right)
end_variable
begin_variable
var2
-1
2
Atom holding(robot, bowl)
Atom on(bowl, shelf)
end_variable
0
begin_state
0
0
1
end_state
begin_goal
1
2 1
end_goal
4
begin_operator
pick-up robot left bowl shelf
0
2
0 0 0 1
0 2 1 0
1
end_operator
begin_operator
pick-up robot right bowl shelf
0
2
0 1 0 1
0 2 1 0
1
end_operator
begin_operator
put-down robot left bowl shelf
0
2
0 0 -1 0
0 2 0 1
1
end_operator
begin_operator
put-down robot right bowl shelf
0
2
0 1 -1 0
0 2 0 1
1
end_operator
0

В╪6
╤=и=
.
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	АР
A
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
+
Ceil
x"T
y"T"
Ttype:
2
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

└
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

└
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(Р
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
н
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
╘
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
ю
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
М
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
р
ResourceApplyAdam
var
m
v
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
0
Round
x"T
y"T"
Ttype:

2	
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
/
Sign
x"T
y"T"
Ttype:

2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
А
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ
9
VarIsInitializedOp
resource
is_initialized
И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.1.02v2.1.0-rc2-17-ge5bf8deУ╦5
~
PlaceholderPlaceholder*/
_output_shapes
:         *
dtype0*$
shape:         
G
lrPlaceholder*
_output_shapes
:*
dtype0*
shape:
Ж
#intercept/Initializer/initial_valueConst*
_class
loc:@intercept*
_output_shapes
: *
dtype0*
valueB
 *╖╤8
Х
	interceptVarHandleOp*
_class
loc:@intercept*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_name	intercept
c
*intercept/IsInitialized/VarIsInitializedOpVarIsInitializedOp	intercept*
_output_shapes
: 
a
intercept/AssignAssignVariableOp	intercept#intercept/Initializer/initial_value*
dtype0
_
intercept/Read/ReadVariableOpReadVariableOp	intercept*
_output_shapes
: *
dtype0
~
slope/Initializer/initial_valueConst*
_class

loc:@slope*
_output_shapes
: *
dtype0*
valueB
 *  А?
Й
slopeVarHandleOp*
_class

loc:@slope*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameslope
[
&slope/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope*
_output_shapes
: 
U
slope/AssignAssignVariableOpslopeslope/Initializer/initial_value*
dtype0
W
slope/Read/ReadVariableOpReadVariableOpslope*
_output_shapes
: *
dtype0
К
%intercept_1/Initializer/initial_valueConst*
_class
loc:@intercept_1*
_output_shapes
: *
dtype0*
valueB
 *╖╤8
Ы
intercept_1VarHandleOp*
_class
loc:@intercept_1*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameintercept_1
g
,intercept_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_1*
_output_shapes
: 
g
intercept_1/AssignAssignVariableOpintercept_1%intercept_1/Initializer/initial_value*
dtype0
c
intercept_1/Read/ReadVariableOpReadVariableOpintercept_1*
_output_shapes
: *
dtype0
В
!slope_1/Initializer/initial_valueConst*
_class
loc:@slope_1*
_output_shapes
: *
dtype0*
valueB
 *  А?
П
slope_1VarHandleOp*
_class
loc:@slope_1*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_name	slope_1
_
(slope_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_1*
_output_shapes
: 
[
slope_1/AssignAssignVariableOpslope_1!slope_1/Initializer/initial_value*
dtype0
[
slope_1/Read/ReadVariableOpReadVariableOpslope_1*
_output_shapes
: *
dtype0
К
%intercept_2/Initializer/initial_valueConst*
_class
loc:@intercept_2*
_output_shapes
: *
dtype0*
valueB
 *╖╤8
Ы
intercept_2VarHandleOp*
_class
loc:@intercept_2*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameintercept_2
g
,intercept_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_2*
_output_shapes
: 
g
intercept_2/AssignAssignVariableOpintercept_2%intercept_2/Initializer/initial_value*
dtype0
c
intercept_2/Read/ReadVariableOpReadVariableOpintercept_2*
_output_shapes
: *
dtype0
В
!slope_2/Initializer/initial_valueConst*
_class
loc:@slope_2*
_output_shapes
: *
dtype0*
valueB
 *  А?
П
slope_2VarHandleOp*
_class
loc:@slope_2*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_name	slope_2
_
(slope_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_2*
_output_shapes
: 
[
slope_2/AssignAssignVariableOpslope_2!slope_2/Initializer/initial_value*
dtype0
[
slope_2/Read/ReadVariableOpReadVariableOpslope_2*
_output_shapes
: *
dtype0
К
%intercept_3/Initializer/initial_valueConst*
_class
loc:@intercept_3*
_output_shapes
: *
dtype0*
valueB
 *╖╤8
Ы
intercept_3VarHandleOp*
_class
loc:@intercept_3*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameintercept_3
g
,intercept_3/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_3*
_output_shapes
: 
g
intercept_3/AssignAssignVariableOpintercept_3%intercept_3/Initializer/initial_value*
dtype0
c
intercept_3/Read/ReadVariableOpReadVariableOpintercept_3*
_output_shapes
: *
dtype0
В
!slope_3/Initializer/initial_valueConst*
_class
loc:@slope_3*
_output_shapes
: *
dtype0*
valueB
 *  А?
П
slope_3VarHandleOp*
_class
loc:@slope_3*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_name	slope_3
_
(slope_3/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_3*
_output_shapes
: 
[
slope_3/AssignAssignVariableOpslope_3!slope_3/Initializer/initial_value*
dtype0
[
slope_3/Read/ReadVariableOpReadVariableOpslope_3*
_output_shapes
: *
dtype0
К
%intercept_4/Initializer/initial_valueConst*
_class
loc:@intercept_4*
_output_shapes
: *
dtype0*
valueB
 *╖╤8
Ы
intercept_4VarHandleOp*
_class
loc:@intercept_4*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameintercept_4
g
,intercept_4/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_4*
_output_shapes
: 
g
intercept_4/AssignAssignVariableOpintercept_4%intercept_4/Initializer/initial_value*
dtype0
c
intercept_4/Read/ReadVariableOpReadVariableOpintercept_4*
_output_shapes
: *
dtype0
В
!slope_4/Initializer/initial_valueConst*
_class
loc:@slope_4*
_output_shapes
: *
dtype0*
valueB
 *  А?
П
slope_4VarHandleOp*
_class
loc:@slope_4*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_name	slope_4
_
(slope_4/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_4*
_output_shapes
: 
[
slope_4/AssignAssignVariableOpslope_4!slope_4/Initializer/initial_value*
dtype0
[
slope_4/Read/ReadVariableOpReadVariableOpslope_4*
_output_shapes
: *
dtype0
К
%intercept_5/Initializer/initial_valueConst*
_class
loc:@intercept_5*
_output_shapes
: *
dtype0*
valueB
 *╖╤8
Ы
intercept_5VarHandleOp*
_class
loc:@intercept_5*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameintercept_5
g
,intercept_5/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_5*
_output_shapes
: 
g
intercept_5/AssignAssignVariableOpintercept_5%intercept_5/Initializer/initial_value*
dtype0
c
intercept_5/Read/ReadVariableOpReadVariableOpintercept_5*
_output_shapes
: *
dtype0
В
!slope_5/Initializer/initial_valueConst*
_class
loc:@slope_5*
_output_shapes
: *
dtype0*
valueB
 *  А?
П
slope_5VarHandleOp*
_class
loc:@slope_5*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_name	slope_5
_
(slope_5/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_5*
_output_shapes
: 
[
slope_5/AssignAssignVariableOpslope_5!slope_5/Initializer/initial_value*
dtype0
[
slope_5/Read/ReadVariableOpReadVariableOpslope_5*
_output_shapes
: *
dtype0
f
activation/IdentityIdentityPlaceholder*
T0*/
_output_shapes
:         
T
activation/ConstConst*
_output_shapes
: *
dtype0*
valueB	 Bgtc
л
0conv2d/kernel/Initializer/truncated_normal/shapeConst* 
_class
loc:@conv2d/kernel*
_output_shapes
:*
dtype0*%
valueB"            
Ц
/conv2d/kernel/Initializer/truncated_normal/meanConst* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ш
1conv2d/kernel/Initializer/truncated_normal/stddevConst* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
dtype0*
valueB
 *▄ц°=
°
:conv2d/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal0conv2d/kernel/Initializer/truncated_normal/shape*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:*
dtype0*

seed *
seed2 
ў
.conv2d/kernel/Initializer/truncated_normal/mulMul:conv2d/kernel/Initializer/truncated_normal/TruncatedNormal1conv2d/kernel/Initializer/truncated_normal/stddev*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:
х
*conv2d/kernel/Initializer/truncated_normalAdd.conv2d/kernel/Initializer/truncated_normal/mul/conv2d/kernel/Initializer/truncated_normal/mean*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:
▒
conv2d/kernelVarHandleOp* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
	container *
dtype0*
shape:*
shared_nameconv2d/kernel
k
.conv2d/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/kernel*
_output_shapes
: 
p
conv2d/kernel/AssignAssignVariableOpconv2d/kernel*conv2d/kernel/Initializer/truncated_normal*
dtype0
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
e
conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
o
conv2d/Abs/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
]

conv2d/AbsAbsconv2d/Abs/ReadVariableOp*
T0*&
_output_shapes
:
Q
conv2d/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
^

conv2d/addAddV2
conv2d/Absconv2d/add/y*
T0*&
_output_shapes
:
N

conv2d/LogLog
conv2d/add*
T0*&
_output_shapes
:
Q
conv2d/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  @
B
conv2d/Log_1Logconv2d/Const*
T0*
_output_shapes
: 
d
conv2d/truedivRealDiv
conv2d/Logconv2d/Log_1*
T0*&
_output_shapes
:
S
conv2d/ReadVariableOpReadVariableOpslope*
_output_shapes
: *
dtype0
i

conv2d/mulMulconv2d/ReadVariableOpconv2d/truediv*
T0*&
_output_shapes
:
Y
conv2d/ReadVariableOp_1ReadVariableOp	intercept*
_output_shapes
: *
dtype0
k
conv2d/add_1AddV2conv2d/ReadVariableOp_1
conv2d/mul*
T0*&
_output_shapes
:
Д
conv2d/differentiable_roundRoundconv2d/add_1*
T0*
_gradient_op_type
Identity*&
_output_shapes
:
x
"conv2d/GreaterEqual/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
Z
conv2d/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
П
conv2d/GreaterEqualGreaterEqual"conv2d/GreaterEqual/ReadVariableOpconv2d/GreaterEqual/y*
T0*&
_output_shapes
:
u
conv2d/ones_like/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
o
conv2d/ones_like/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
[
conv2d/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Л
conv2d/ones_likeFillconv2d/ones_like/Shapeconv2d/ones_like/Const*
T0*&
_output_shapes
:*

index_type0
v
conv2d/zeros_likeConst*&
_output_shapes
:*
dtype0*%
valueB*    
Ж
conv2d/SelectV2SelectV2conv2d/GreaterEqualconv2d/ones_likeconv2d/zeros_like*
T0*&
_output_shapes
:
u
conv2d/LessEqual/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
W
conv2d/LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
Г
conv2d/LessEqual	LessEqualconv2d/LessEqual/ReadVariableOpconv2d/LessEqual/y*
T0*&
_output_shapes
:
w
!conv2d/ones_like_1/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
q
conv2d/ones_like_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
]
conv2d/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
С
conv2d/ones_like_1Fillconv2d/ones_like_1/Shapeconv2d/ones_like_1/Const*
T0*&
_output_shapes
:*

index_type0
S
conv2d/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
h
conv2d/mul_1Mulconv2d/mul_1/xconv2d/ones_like_1*
T0*&
_output_shapes
:
x
conv2d/zeros_like_1Const*&
_output_shapes
:*
dtype0*%
valueB*    
Г
conv2d/SelectV2_1SelectV2conv2d/LessEqualconv2d/mul_1conv2d/zeros_like_1*
T0*&
_output_shapes
:
m
conv2d/lp_weightsAddconv2d/SelectV2_1conv2d/SelectV2*
T0*&
_output_shapes
:
Q
conv2d/pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
m

conv2d/powPowconv2d/pow/xconv2d/differentiable_round*
T0*&
_output_shapes
:
c
conv2d/mul_2Mulconv2d/lp_weights
conv2d/pow*
T0*&
_output_shapes
:
r
conv2d/Conv2D/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
Г
conv2d/Conv2DConv2DPlaceholderconv2d/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
¤
conv2d/Conv2D_1Conv2Dactivation/Identityconv2d/mul_2*
T0*/
_output_shapes
:         *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
R
conv2d/Const_1Const*
_output_shapes
: *
dtype0*
valueB	 Bgtc
b
activation_1/ReluReluconv2d/Conv2D*
T0*/
_output_shapes
:         
f
activation_1/Relu_1Reluconv2d/Conv2D_1*
T0*/
_output_shapes
:         
p
activation_1/IdentityIdentityactivation_1/Relu_1*
T0*/
_output_shapes
:         
V
activation_1/ConstConst*
_output_shapes
: *
dtype0*
valueB	 Bgtc
┐
max_pooling2d/MaxPoolMaxPoolactivation_1/Relu*
T0*/
_output_shapes
:         *
data_formatNHWC*
ksize
*
paddingSAME*
strides

┼
max_pooling2d/MaxPool_1MaxPoolactivation_1/Identity*
T0*/
_output_shapes
:         *
data_formatNHWC*
ksize
*
paddingSAME*
strides

u
max_pooling2d/IdentityIdentitymax_pooling2d/MaxPool_1*
T0*/
_output_shapes
:         
W
max_pooling2d/ConstConst*
_output_shapes
: *
dtype0*
valueB	 Bgtc
п
2conv2d_1/kernel/Initializer/truncated_normal/shapeConst*"
_class
loc:@conv2d_1/kernel*
_output_shapes
:*
dtype0*%
valueB"         
   
Ъ
1conv2d_1/kernel/Initializer/truncated_normal/meanConst*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ь
3conv2d_1/kernel/Initializer/truncated_normal/stddevConst*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *вд=
■
<conv2d_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal2conv2d_1/kernel/Initializer/truncated_normal/shape*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
*
dtype0*

seed *
seed2 
 
0conv2d_1/kernel/Initializer/truncated_normal/mulMul<conv2d_1/kernel/Initializer/truncated_normal/TruncatedNormal3conv2d_1/kernel/Initializer/truncated_normal/stddev*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:

э
,conv2d_1/kernel/Initializer/truncated_normalAdd0conv2d_1/kernel/Initializer/truncated_normal/mul1conv2d_1/kernel/Initializer/truncated_normal/mean*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:

╖
conv2d_1/kernelVarHandleOp*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
	container *
dtype0*
shape:
* 
shared_nameconv2d_1/kernel
o
0conv2d_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
v
conv2d_1/kernel/AssignAssignVariableOpconv2d_1/kernel,conv2d_1/kernel/Initializer/truncated_normal*
dtype0
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
g
conv2d_1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
s
conv2d_1/Abs/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
a
conv2d_1/AbsAbsconv2d_1/Abs/ReadVariableOp*
T0*&
_output_shapes
:

S
conv2d_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
d
conv2d_1/addAddV2conv2d_1/Absconv2d_1/add/y*
T0*&
_output_shapes
:

R
conv2d_1/LogLogconv2d_1/add*
T0*&
_output_shapes
:

S
conv2d_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  @
F
conv2d_1/Log_1Logconv2d_1/Const*
T0*
_output_shapes
: 
j
conv2d_1/truedivRealDivconv2d_1/Logconv2d_1/Log_1*
T0*&
_output_shapes
:

W
conv2d_1/ReadVariableOpReadVariableOpslope_1*
_output_shapes
: *
dtype0
o
conv2d_1/mulMulconv2d_1/ReadVariableOpconv2d_1/truediv*
T0*&
_output_shapes
:

]
conv2d_1/ReadVariableOp_1ReadVariableOpintercept_1*
_output_shapes
: *
dtype0
q
conv2d_1/add_1AddV2conv2d_1/ReadVariableOp_1conv2d_1/mul*
T0*&
_output_shapes
:

И
conv2d_1/differentiable_roundRoundconv2d_1/add_1*
T0*
_gradient_op_type
Identity*&
_output_shapes
:

|
$conv2d_1/GreaterEqual/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
\
conv2d_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Х
conv2d_1/GreaterEqualGreaterEqual$conv2d_1/GreaterEqual/ReadVariableOpconv2d_1/GreaterEqual/y*
T0*&
_output_shapes
:

y
!conv2d_1/ones_like/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
q
conv2d_1/ones_like/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
]
conv2d_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
С
conv2d_1/ones_likeFillconv2d_1/ones_like/Shapeconv2d_1/ones_like/Const*
T0*&
_output_shapes
:
*

index_type0
|
#conv2d_1/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"         
   
^
conv2d_1/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Ю
conv2d_1/zeros_likeFill#conv2d_1/zeros_like/shape_as_tensorconv2d_1/zeros_like/Const*
T0*&
_output_shapes
:
*

index_type0
О
conv2d_1/SelectV2SelectV2conv2d_1/GreaterEqualconv2d_1/ones_likeconv2d_1/zeros_like*
T0*&
_output_shapes
:

y
!conv2d_1/LessEqual/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
Y
conv2d_1/LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
Й
conv2d_1/LessEqual	LessEqual!conv2d_1/LessEqual/ReadVariableOpconv2d_1/LessEqual/y*
T0*&
_output_shapes
:

{
#conv2d_1/ones_like_1/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
s
conv2d_1/ones_like_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
_
conv2d_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ч
conv2d_1/ones_like_1Fillconv2d_1/ones_like_1/Shapeconv2d_1/ones_like_1/Const*
T0*&
_output_shapes
:
*

index_type0
U
conv2d_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
n
conv2d_1/mul_1Mulconv2d_1/mul_1/xconv2d_1/ones_like_1*
T0*&
_output_shapes
:

~
%conv2d_1/zeros_like_1/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"         
   
`
conv2d_1/zeros_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
д
conv2d_1/zeros_like_1Fill%conv2d_1/zeros_like_1/shape_as_tensorconv2d_1/zeros_like_1/Const*
T0*&
_output_shapes
:
*

index_type0
Л
conv2d_1/SelectV2_1SelectV2conv2d_1/LessEqualconv2d_1/mul_1conv2d_1/zeros_like_1*
T0*&
_output_shapes
:

s
conv2d_1/lp_weightsAddconv2d_1/SelectV2_1conv2d_1/SelectV2*
T0*&
_output_shapes
:

S
conv2d_1/pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
s
conv2d_1/powPowconv2d_1/pow/xconv2d_1/differentiable_round*
T0*&
_output_shapes
:

i
conv2d_1/mul_2Mulconv2d_1/lp_weightsconv2d_1/pow*
T0*&
_output_shapes
:

v
conv2d_1/Conv2D/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
С
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPoolconv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         
*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
Д
conv2d_1/Conv2D_1Conv2Dmax_pooling2d/Identityconv2d_1/mul_2*
T0*/
_output_shapes
:         
*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
T
conv2d_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB	 Bgtc
d
activation_2/ReluReluconv2d_1/Conv2D*
T0*/
_output_shapes
:         

h
activation_2/Relu_1Reluconv2d_1/Conv2D_1*
T0*/
_output_shapes
:         

p
activation_2/IdentityIdentityactivation_2/Relu_1*
T0*/
_output_shapes
:         

V
activation_2/ConstConst*
_output_shapes
: *
dtype0*
valueB	 Bgtc
┴
max_pooling2d_1/MaxPoolMaxPoolactivation_2/Relu*
T0*/
_output_shapes
:         
*
data_formatNHWC*
ksize
*
paddingSAME*
strides

╟
max_pooling2d_1/MaxPool_1MaxPoolactivation_2/Identity*
T0*/
_output_shapes
:         
*
data_formatNHWC*
ksize
*
paddingSAME*
strides

y
max_pooling2d_1/IdentityIdentitymax_pooling2d_1/MaxPool_1*
T0*/
_output_shapes
:         

Y
max_pooling2d_1/ConstConst*
_output_shapes
: *
dtype0*
valueB	 Bgtc
f
flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ъ  
Л
flatten/ReshapeReshapemax_pooling2d_1/MaxPoolflatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:         ъ
h
flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ъ  
Р
flatten/Reshape_1Reshapemax_pooling2d_1/Identityflatten/Reshape_1/shape*
T0*
Tshape0*(
_output_shapes
:         ъ
Q
flatten/ConstConst*
_output_shapes
: *
dtype0*
valueB	 Bgtc
f
reshape/reshape_1/ShapeShapeflatten/Reshape*
T0*
_output_shapes
:*
out_type0
o
%reshape/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
q
'reshape/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
q
'reshape/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
╙
reshape/reshape_1/strided_sliceStridedSlicereshape/reshape_1/Shape%reshape/reshape_1/strided_slice/stack'reshape/reshape_1/strided_slice/stack_1'reshape/reshape_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
c
!reshape/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :1
c
!reshape/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

c
!reshape/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
ы
reshape/reshape_1/Reshape/shapePackreshape/reshape_1/strided_slice!reshape/reshape_1/Reshape/shape/1!reshape/reshape_1/Reshape/shape/2!reshape/reshape_1/Reshape/shape/3*
N*
T0*
_output_shapes
:*

axis 
Ю
reshape/reshape_1/ReshapeReshapeflatten/Reshapereshape/reshape_1/Reshape/shape*
T0*
Tshape0*/
_output_shapes
:         1

n
reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    1   
      
К
reshape/ReshapeReshapeflatten/Reshapereshape/Reshape/shape*
T0*
Tshape0*/
_output_shapes
:         1

p
reshape/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    1   
      
Р
reshape/Reshape_2Reshapeflatten/Reshape_1reshape/Reshape_2/shape*
T0*
Tshape0*/
_output_shapes
:         1

s
reshape/reshape_requantizeIdentityreshape/Reshape_2*
T0*/
_output_shapes
:         1

Q
reshape/ConstConst*
_output_shapes
: *
dtype0*
valueB	 Bgtc
п
2conv2d_2/kernel/Initializer/truncated_normal/shapeConst*"
_class
loc:@conv2d_2/kernel*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
Ъ
1conv2d_2/kernel/Initializer/truncated_normal/meanConst*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ь
3conv2d_2/kernel/Initializer/truncated_normal/stddevConst*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *!Л╤;
 
<conv2d_2/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal2conv2d_2/kernel/Initializer/truncated_normal/shape*
T0*"
_class
loc:@conv2d_2/kernel*'
_output_shapes
:1
А*
dtype0*

seed *
seed2 
А
0conv2d_2/kernel/Initializer/truncated_normal/mulMul<conv2d_2/kernel/Initializer/truncated_normal/TruncatedNormal3conv2d_2/kernel/Initializer/truncated_normal/stddev*
T0*"
_class
loc:@conv2d_2/kernel*'
_output_shapes
:1
А
ю
,conv2d_2/kernel/Initializer/truncated_normalAdd0conv2d_2/kernel/Initializer/truncated_normal/mul1conv2d_2/kernel/Initializer/truncated_normal/mean*
T0*"
_class
loc:@conv2d_2/kernel*'
_output_shapes
:1
А
╕
conv2d_2/kernelVarHandleOp*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: *
	container *
dtype0*
shape:1
А* 
shared_nameconv2d_2/kernel
o
0conv2d_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
v
conv2d_2/kernel/AssignAssignVariableOpconv2d_2/kernel,conv2d_2/kernel/Initializer/truncated_normal*
dtype0
|
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
g
conv2d_2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
t
conv2d_2/Abs/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
b
conv2d_2/AbsAbsconv2d_2/Abs/ReadVariableOp*
T0*'
_output_shapes
:1
А
S
conv2d_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
e
conv2d_2/addAddV2conv2d_2/Absconv2d_2/add/y*
T0*'
_output_shapes
:1
А
S
conv2d_2/LogLogconv2d_2/add*
T0*'
_output_shapes
:1
А
S
conv2d_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  @
F
conv2d_2/Log_1Logconv2d_2/Const*
T0*
_output_shapes
: 
k
conv2d_2/truedivRealDivconv2d_2/Logconv2d_2/Log_1*
T0*'
_output_shapes
:1
А
W
conv2d_2/ReadVariableOpReadVariableOpslope_2*
_output_shapes
: *
dtype0
p
conv2d_2/mulMulconv2d_2/ReadVariableOpconv2d_2/truediv*
T0*'
_output_shapes
:1
А
]
conv2d_2/ReadVariableOp_1ReadVariableOpintercept_2*
_output_shapes
: *
dtype0
r
conv2d_2/add_1AddV2conv2d_2/ReadVariableOp_1conv2d_2/mul*
T0*'
_output_shapes
:1
А
Й
conv2d_2/differentiable_roundRoundconv2d_2/add_1*
T0*
_gradient_op_type
Identity*'
_output_shapes
:1
А
}
$conv2d_2/GreaterEqual/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
\
conv2d_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Ц
conv2d_2/GreaterEqualGreaterEqual$conv2d_2/GreaterEqual/ReadVariableOpconv2d_2/GreaterEqual/y*
T0*'
_output_shapes
:1
А
z
!conv2d_2/ones_like/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
q
conv2d_2/ones_like/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
]
conv2d_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Т
conv2d_2/ones_likeFillconv2d_2/ones_like/Shapeconv2d_2/ones_like/Const*
T0*'
_output_shapes
:1
А*

index_type0
|
#conv2d_2/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
^
conv2d_2/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Я
conv2d_2/zeros_likeFill#conv2d_2/zeros_like/shape_as_tensorconv2d_2/zeros_like/Const*
T0*'
_output_shapes
:1
А*

index_type0
П
conv2d_2/SelectV2SelectV2conv2d_2/GreaterEqualconv2d_2/ones_likeconv2d_2/zeros_like*
T0*'
_output_shapes
:1
А
z
!conv2d_2/LessEqual/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
Y
conv2d_2/LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
К
conv2d_2/LessEqual	LessEqual!conv2d_2/LessEqual/ReadVariableOpconv2d_2/LessEqual/y*
T0*'
_output_shapes
:1
А
|
#conv2d_2/ones_like_1/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
s
conv2d_2/ones_like_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
_
conv2d_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ш
conv2d_2/ones_like_1Fillconv2d_2/ones_like_1/Shapeconv2d_2/ones_like_1/Const*
T0*'
_output_shapes
:1
А*

index_type0
U
conv2d_2/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
o
conv2d_2/mul_1Mulconv2d_2/mul_1/xconv2d_2/ones_like_1*
T0*'
_output_shapes
:1
А
~
%conv2d_2/zeros_like_1/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
`
conv2d_2/zeros_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
е
conv2d_2/zeros_like_1Fill%conv2d_2/zeros_like_1/shape_as_tensorconv2d_2/zeros_like_1/Const*
T0*'
_output_shapes
:1
А*

index_type0
М
conv2d_2/SelectV2_1SelectV2conv2d_2/LessEqualconv2d_2/mul_1conv2d_2/zeros_like_1*
T0*'
_output_shapes
:1
А
t
conv2d_2/lp_weightsAddconv2d_2/SelectV2_1conv2d_2/SelectV2*
T0*'
_output_shapes
:1
А
S
conv2d_2/pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
t
conv2d_2/powPowconv2d_2/pow/xconv2d_2/differentiable_round*
T0*'
_output_shapes
:1
А
j
conv2d_2/mul_2Mulconv2d_2/lp_weightsconv2d_2/pow*
T0*'
_output_shapes
:1
А
w
conv2d_2/Conv2D/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
Н
conv2d_2/Conv2DConv2Dreshape/Reshapeconv2d_2/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
К
conv2d_2/Conv2D_1Conv2Dreshape/reshape_requantizeconv2d_2/mul_2*
T0*0
_output_shapes
:         А*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
T
conv2d_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB	 Bgtc
e
activation_3/ReluReluconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
i
activation_3/Relu_1Reluconv2d_2/Conv2D_1*
T0*0
_output_shapes
:         А
g
activation_3/AbsAbsactivation_3/Relu_1*
T0*0
_output_shapes
:         А
W
activation_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
z
activation_3/addAddV2activation_3/Absactivation_3/add/y*
T0*0
_output_shapes
:         А
d
activation_3/LogLogactivation_3/add*
T0*0
_output_shapes
:         А
W
activation_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  @
N
activation_3/Log_1Logactivation_3/Const*
T0*
_output_shapes
: 
А
activation_3/truedivRealDivactivation_3/Logactivation_3/Log_1*
T0*0
_output_shapes
:         А
[
activation_3/ReadVariableOpReadVariableOpslope_3*
_output_shapes
: *
dtype0
Е
activation_3/mulMulactivation_3/ReadVariableOpactivation_3/truediv*
T0*0
_output_shapes
:         А
a
activation_3/ReadVariableOp_1ReadVariableOpintercept_3*
_output_shapes
: *
dtype0
З
activation_3/add_1AddV2activation_3/ReadVariableOp_1activation_3/mul*
T0*0
_output_shapes
:         А
Ъ
!activation_3/differentiable_roundRoundactivation_3/add_1*
T0*
_gradient_op_type
Identity*0
_output_shapes
:         А
`
activation_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Ц
activation_3/GreaterEqualGreaterEqualactivation_3/Relu_1activation_3/GreaterEqual/y*
T0*0
_output_shapes
:         А
o
activation_3/ones_like/ShapeShapeactivation_3/Relu_1*
T0*
_output_shapes
:*
out_type0
a
activation_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
з
activation_3/ones_likeFillactivation_3/ones_like/Shapeactivation_3/ones_like/Const*
T0*0
_output_shapes
:         А*

index_type0
t
activation_3/zeros_like	ZerosLikeactivation_3/Relu_1*
T0*0
_output_shapes
:         А
и
activation_3/SelectV2SelectV2activation_3/GreaterEqualactivation_3/ones_likeactivation_3/zeros_like*
T0*0
_output_shapes
:         А
]
activation_3/LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
Н
activation_3/LessEqual	LessEqualactivation_3/Relu_1activation_3/LessEqual/y*
T0*0
_output_shapes
:         А
q
activation_3/ones_like_1/ShapeShapeactivation_3/Relu_1*
T0*
_output_shapes
:*
out_type0
c
activation_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
н
activation_3/ones_like_1Fillactivation_3/ones_like_1/Shapeactivation_3/ones_like_1/Const*
T0*0
_output_shapes
:         А*

index_type0
Y
activation_3/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
Д
activation_3/mul_1Mulactivation_3/mul_1/xactivation_3/ones_like_1*
T0*0
_output_shapes
:         А
v
activation_3/zeros_like_1	ZerosLikeactivation_3/Relu_1*
T0*0
_output_shapes
:         А
е
activation_3/SelectV2_1SelectV2activation_3/LessEqualactivation_3/mul_1activation_3/zeros_like_1*
T0*0
_output_shapes
:         А
Д
activation_3/Add_2Addactivation_3/SelectV2_1activation_3/SelectV2*
T0*0
_output_shapes
:         А
W
activation_3/pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
Й
activation_3/powPowactivation_3/pow/x!activation_3/differentiable_round*
T0*0
_output_shapes
:         А
z
activation_3/mul_2Mulactivation_3/Add_2activation_3/pow*
T0*0
_output_shapes
:         А
X
activation_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB	 Bgtc
j
reshape_1/reshape_2/ShapeShapeactivation_3/Relu*
T0*
_output_shapes
:*
out_type0
q
'reshape_1/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
s
)reshape_1/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
s
)reshape_1/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
▌
!reshape_1/reshape_2/strided_sliceStridedSlicereshape_1/reshape_2/Shape'reshape_1/reshape_2/strided_slice/stack)reshape_1/reshape_2/strided_slice/stack_1)reshape_1/reshape_2/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
f
#reshape_1/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :А
e
#reshape_1/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
e
#reshape_1/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
ї
!reshape_1/reshape_2/Reshape/shapePack!reshape_1/reshape_2/strided_slice#reshape_1/reshape_2/Reshape/shape/1#reshape_1/reshape_2/Reshape/shape/2#reshape_1/reshape_2/Reshape/shape/3*
N*
T0*
_output_shapes
:*

axis 
е
reshape_1/reshape_2/ReshapeReshapeactivation_3/Relu!reshape_1/reshape_2/Reshape/shape*
T0*
Tshape0*0
_output_shapes
:         А
p
reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    А         
С
reshape_1/ReshapeReshapeactivation_3/Relureshape_1/Reshape/shape*
T0*
Tshape0*0
_output_shapes
:         А
r
reshape_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    А         
Ц
reshape_1/Reshape_1Reshapeactivation_3/mul_2reshape_1/Reshape_1/shape*
T0*
Tshape0*0
_output_shapes
:         А
z
reshape_1/reshape_1_requantizeIdentityreshape_1/Reshape_1*
T0*0
_output_shapes
:         А
S
reshape_1/ConstConst*
_output_shapes
: *
dtype0*
valueB	 Bgtc
п
2conv2d_3/kernel/Initializer/truncated_normal/shapeConst*"
_class
loc:@conv2d_3/kernel*
_output_shapes
:*
dtype0*%
valueB"А         А   
Ъ
1conv2d_3/kernel/Initializer/truncated_normal/meanConst*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ь
3conv2d_3/kernel/Initializer/truncated_normal/stddevConst*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *¤¤L<
А
<conv2d_3/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal2conv2d_3/kernel/Initializer/truncated_normal/shape*
T0*"
_class
loc:@conv2d_3/kernel*(
_output_shapes
:АА*
dtype0*

seed *
seed2 
Б
0conv2d_3/kernel/Initializer/truncated_normal/mulMul<conv2d_3/kernel/Initializer/truncated_normal/TruncatedNormal3conv2d_3/kernel/Initializer/truncated_normal/stddev*
T0*"
_class
loc:@conv2d_3/kernel*(
_output_shapes
:АА
я
,conv2d_3/kernel/Initializer/truncated_normalAdd0conv2d_3/kernel/Initializer/truncated_normal/mul1conv2d_3/kernel/Initializer/truncated_normal/mean*
T0*"
_class
loc:@conv2d_3/kernel*(
_output_shapes
:АА
╣
conv2d_3/kernelVarHandleOp*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: *
	container *
dtype0*
shape:АА* 
shared_nameconv2d_3/kernel
o
0conv2d_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 
v
conv2d_3/kernel/AssignAssignVariableOpconv2d_3/kernel,conv2d_3/kernel/Initializer/truncated_normal*
dtype0
}
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
g
conv2d_3/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
u
conv2d_3/Abs/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
c
conv2d_3/AbsAbsconv2d_3/Abs/ReadVariableOp*
T0*(
_output_shapes
:АА
S
conv2d_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
f
conv2d_3/addAddV2conv2d_3/Absconv2d_3/add/y*
T0*(
_output_shapes
:АА
T
conv2d_3/LogLogconv2d_3/add*
T0*(
_output_shapes
:АА
S
conv2d_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  @
F
conv2d_3/Log_1Logconv2d_3/Const*
T0*
_output_shapes
: 
l
conv2d_3/truedivRealDivconv2d_3/Logconv2d_3/Log_1*
T0*(
_output_shapes
:АА
W
conv2d_3/ReadVariableOpReadVariableOpslope_4*
_output_shapes
: *
dtype0
q
conv2d_3/mulMulconv2d_3/ReadVariableOpconv2d_3/truediv*
T0*(
_output_shapes
:АА
]
conv2d_3/ReadVariableOp_1ReadVariableOpintercept_4*
_output_shapes
: *
dtype0
s
conv2d_3/add_1AddV2conv2d_3/ReadVariableOp_1conv2d_3/mul*
T0*(
_output_shapes
:АА
К
conv2d_3/differentiable_roundRoundconv2d_3/add_1*
T0*
_gradient_op_type
Identity*(
_output_shapes
:АА
~
$conv2d_3/GreaterEqual/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
\
conv2d_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Ч
conv2d_3/GreaterEqualGreaterEqual$conv2d_3/GreaterEqual/ReadVariableOpconv2d_3/GreaterEqual/y*
T0*(
_output_shapes
:АА
{
!conv2d_3/ones_like/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
q
conv2d_3/ones_like/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
]
conv2d_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
У
conv2d_3/ones_likeFillconv2d_3/ones_like/Shapeconv2d_3/ones_like/Const*
T0*(
_output_shapes
:АА*

index_type0
|
#conv2d_3/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
^
conv2d_3/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
а
conv2d_3/zeros_likeFill#conv2d_3/zeros_like/shape_as_tensorconv2d_3/zeros_like/Const*
T0*(
_output_shapes
:АА*

index_type0
Р
conv2d_3/SelectV2SelectV2conv2d_3/GreaterEqualconv2d_3/ones_likeconv2d_3/zeros_like*
T0*(
_output_shapes
:АА
{
!conv2d_3/LessEqual/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
Y
conv2d_3/LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
Л
conv2d_3/LessEqual	LessEqual!conv2d_3/LessEqual/ReadVariableOpconv2d_3/LessEqual/y*
T0*(
_output_shapes
:АА
}
#conv2d_3/ones_like_1/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
s
conv2d_3/ones_like_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
_
conv2d_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Щ
conv2d_3/ones_like_1Fillconv2d_3/ones_like_1/Shapeconv2d_3/ones_like_1/Const*
T0*(
_output_shapes
:АА*

index_type0
U
conv2d_3/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
p
conv2d_3/mul_1Mulconv2d_3/mul_1/xconv2d_3/ones_like_1*
T0*(
_output_shapes
:АА
~
%conv2d_3/zeros_like_1/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
`
conv2d_3/zeros_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
ж
conv2d_3/zeros_like_1Fill%conv2d_3/zeros_like_1/shape_as_tensorconv2d_3/zeros_like_1/Const*
T0*(
_output_shapes
:АА*

index_type0
Н
conv2d_3/SelectV2_1SelectV2conv2d_3/LessEqualconv2d_3/mul_1conv2d_3/zeros_like_1*
T0*(
_output_shapes
:АА
u
conv2d_3/lp_weightsAddconv2d_3/SelectV2_1conv2d_3/SelectV2*
T0*(
_output_shapes
:АА
S
conv2d_3/pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
u
conv2d_3/powPowconv2d_3/pow/xconv2d_3/differentiable_round*
T0*(
_output_shapes
:АА
k
conv2d_3/mul_2Mulconv2d_3/lp_weightsconv2d_3/pow*
T0*(
_output_shapes
:АА
x
conv2d_3/Conv2D/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
П
conv2d_3/Conv2DConv2Dreshape_1/Reshapeconv2d_3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
О
conv2d_3/Conv2D_1Conv2Dreshape_1/reshape_1_requantizeconv2d_3/mul_2*
T0*0
_output_shapes
:         А*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
T
conv2d_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB	 Bgtc
e
activation_4/ReluReluconv2d_3/Conv2D*
T0*0
_output_shapes
:         А
i
activation_4/Relu_1Reluconv2d_3/Conv2D_1*
T0*0
_output_shapes
:         А
q
activation_4/IdentityIdentityactivation_4/Relu_1*
T0*0
_output_shapes
:         А
V
activation_4/ConstConst*
_output_shapes
: *
dtype0*
valueB	 Bgtc
j
reshape_2/reshape_3/ShapeShapeactivation_4/Relu*
T0*
_output_shapes
:*
out_type0
q
'reshape_2/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
s
)reshape_2/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
s
)reshape_2/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
▌
!reshape_2/reshape_3/strided_sliceStridedSlicereshape_2/reshape_3/Shape'reshape_2/reshape_3/strided_slice/stack)reshape_2/reshape_3/strided_slice/stack_1)reshape_2/reshape_3/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
f
#reshape_2/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :А
e
#reshape_2/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
e
#reshape_2/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
ї
!reshape_2/reshape_3/Reshape/shapePack!reshape_2/reshape_3/strided_slice#reshape_2/reshape_3/Reshape/shape/1#reshape_2/reshape_3/Reshape/shape/2#reshape_2/reshape_3/Reshape/shape/3*
N*
T0*
_output_shapes
:*

axis 
е
reshape_2/reshape_3/ReshapeReshapeactivation_4/Relu!reshape_2/reshape_3/Reshape/shape*
T0*
Tshape0*0
_output_shapes
:         А
p
reshape_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    А         
С
reshape_2/ReshapeReshapeactivation_4/Relureshape_2/Reshape/shape*
T0*
Tshape0*0
_output_shapes
:         А
r
reshape_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    А         
Щ
reshape_2/Reshape_1Reshapeactivation_4/Identityreshape_2/Reshape_1/shape*
T0*
Tshape0*0
_output_shapes
:         А
z
reshape_2/reshape_2_requantizeIdentityreshape_2/Reshape_1*
T0*0
_output_shapes
:         А
S
reshape_2/ConstConst*
_output_shapes
: *
dtype0*
valueB	 Bgtc
п
2conv2d_4/kernel/Initializer/truncated_normal/shapeConst*"
_class
loc:@conv2d_4/kernel*
_output_shapes
:*
dtype0*%
valueB"А         
   
Ъ
1conv2d_4/kernel/Initializer/truncated_normal/meanConst*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ь
3conv2d_4/kernel/Initializer/truncated_normal/stddevConst*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: *
dtype0*
valueB
 *█/=
 
<conv2d_4/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal2conv2d_4/kernel/Initializer/truncated_normal/shape*
T0*"
_class
loc:@conv2d_4/kernel*'
_output_shapes
:А
*
dtype0*

seed *
seed2 
А
0conv2d_4/kernel/Initializer/truncated_normal/mulMul<conv2d_4/kernel/Initializer/truncated_normal/TruncatedNormal3conv2d_4/kernel/Initializer/truncated_normal/stddev*
T0*"
_class
loc:@conv2d_4/kernel*'
_output_shapes
:А

ю
,conv2d_4/kernel/Initializer/truncated_normalAdd0conv2d_4/kernel/Initializer/truncated_normal/mul1conv2d_4/kernel/Initializer/truncated_normal/mean*
T0*"
_class
loc:@conv2d_4/kernel*'
_output_shapes
:А

╕
conv2d_4/kernelVarHandleOp*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: *
	container *
dtype0*
shape:А
* 
shared_nameconv2d_4/kernel
o
0conv2d_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/kernel*
_output_shapes
: 
v
conv2d_4/kernel/AssignAssignVariableOpconv2d_4/kernel,conv2d_4/kernel/Initializer/truncated_normal*
dtype0
|
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
g
conv2d_4/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
t
conv2d_4/Abs/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
b
conv2d_4/AbsAbsconv2d_4/Abs/ReadVariableOp*
T0*'
_output_shapes
:А

S
conv2d_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
e
conv2d_4/addAddV2conv2d_4/Absconv2d_4/add/y*
T0*'
_output_shapes
:А

S
conv2d_4/LogLogconv2d_4/add*
T0*'
_output_shapes
:А

S
conv2d_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  @
F
conv2d_4/Log_1Logconv2d_4/Const*
T0*
_output_shapes
: 
k
conv2d_4/truedivRealDivconv2d_4/Logconv2d_4/Log_1*
T0*'
_output_shapes
:А

W
conv2d_4/ReadVariableOpReadVariableOpslope_5*
_output_shapes
: *
dtype0
p
conv2d_4/mulMulconv2d_4/ReadVariableOpconv2d_4/truediv*
T0*'
_output_shapes
:А

]
conv2d_4/ReadVariableOp_1ReadVariableOpintercept_5*
_output_shapes
: *
dtype0
r
conv2d_4/add_1AddV2conv2d_4/ReadVariableOp_1conv2d_4/mul*
T0*'
_output_shapes
:А

Й
conv2d_4/differentiable_roundRoundconv2d_4/add_1*
T0*
_gradient_op_type
Identity*'
_output_shapes
:А

}
$conv2d_4/GreaterEqual/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
\
conv2d_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Ц
conv2d_4/GreaterEqualGreaterEqual$conv2d_4/GreaterEqual/ReadVariableOpconv2d_4/GreaterEqual/y*
T0*'
_output_shapes
:А

z
!conv2d_4/ones_like/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
q
conv2d_4/ones_like/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
]
conv2d_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Т
conv2d_4/ones_likeFillconv2d_4/ones_like/Shapeconv2d_4/ones_like/Const*
T0*'
_output_shapes
:А
*

index_type0
|
#conv2d_4/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
^
conv2d_4/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Я
conv2d_4/zeros_likeFill#conv2d_4/zeros_like/shape_as_tensorconv2d_4/zeros_like/Const*
T0*'
_output_shapes
:А
*

index_type0
П
conv2d_4/SelectV2SelectV2conv2d_4/GreaterEqualconv2d_4/ones_likeconv2d_4/zeros_like*
T0*'
_output_shapes
:А

z
!conv2d_4/LessEqual/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
Y
conv2d_4/LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
К
conv2d_4/LessEqual	LessEqual!conv2d_4/LessEqual/ReadVariableOpconv2d_4/LessEqual/y*
T0*'
_output_shapes
:А

|
#conv2d_4/ones_like_1/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
s
conv2d_4/ones_like_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
_
conv2d_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ш
conv2d_4/ones_like_1Fillconv2d_4/ones_like_1/Shapeconv2d_4/ones_like_1/Const*
T0*'
_output_shapes
:А
*

index_type0
U
conv2d_4/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
o
conv2d_4/mul_1Mulconv2d_4/mul_1/xconv2d_4/ones_like_1*
T0*'
_output_shapes
:А

~
%conv2d_4/zeros_like_1/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
`
conv2d_4/zeros_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
е
conv2d_4/zeros_like_1Fill%conv2d_4/zeros_like_1/shape_as_tensorconv2d_4/zeros_like_1/Const*
T0*'
_output_shapes
:А
*

index_type0
М
conv2d_4/SelectV2_1SelectV2conv2d_4/LessEqualconv2d_4/mul_1conv2d_4/zeros_like_1*
T0*'
_output_shapes
:А

t
conv2d_4/lp_weightsAddconv2d_4/SelectV2_1conv2d_4/SelectV2*
T0*'
_output_shapes
:А

S
conv2d_4/pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
t
conv2d_4/powPowconv2d_4/pow/xconv2d_4/differentiable_round*
T0*'
_output_shapes
:А

j
conv2d_4/mul_2Mulconv2d_4/lp_weightsconv2d_4/pow*
T0*'
_output_shapes
:А

w
conv2d_4/Conv2D/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
О
conv2d_4/Conv2DConv2Dreshape_2/Reshapeconv2d_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         
*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
Н
conv2d_4/Conv2D_1Conv2Dreshape_2/reshape_2_requantizeconv2d_4/mul_2*
T0*/
_output_shapes
:         
*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
T
conv2d_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB	 Bgtc
h
flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    
   
Ж
flatten_1/ReshapeReshapeconv2d_4/Conv2Dflatten_1/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         

j
flatten_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    
   
М
flatten_1/Reshape_1Reshapeconv2d_4/Conv2D_1flatten_1/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:         

S
flatten_1/ConstConst*
_output_shapes
: *
dtype0*
valueB	 Bgtc
p
Placeholder_1Placeholder*'
_output_shapes
:         
*
dtype0*
shape:         

h
&softmax_cross_entropy_with_logits/RankConst*
_output_shapes
: *
dtype0*
value	B :
x
'softmax_cross_entropy_with_logits/ShapeShapeflatten_1/Reshape*
T0*
_output_shapes
:*
out_type0
j
(softmax_cross_entropy_with_logits/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
z
)softmax_cross_entropy_with_logits/Shape_1Shapeflatten_1/Reshape*
T0*
_output_shapes
:*
out_type0
i
'softmax_cross_entropy_with_logits/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
а
%softmax_cross_entropy_with_logits/SubSub(softmax_cross_entropy_with_logits/Rank_1'softmax_cross_entropy_with_logits/Sub/y*
T0*
_output_shapes
: 
Ц
-softmax_cross_entropy_with_logits/Slice/beginPack%softmax_cross_entropy_with_logits/Sub*
N*
T0*
_output_shapes
:*

axis 
v
,softmax_cross_entropy_with_logits/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
ъ
'softmax_cross_entropy_with_logits/SliceSlice)softmax_cross_entropy_with_logits/Shape_1-softmax_cross_entropy_with_logits/Slice/begin,softmax_cross_entropy_with_logits/Slice/size*
Index0*
T0*
_output_shapes
:
Д
1softmax_cross_entropy_with_logits/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
         
o
-softmax_cross_entropy_with_logits/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
∙
(softmax_cross_entropy_with_logits/concatConcatV21softmax_cross_entropy_with_logits/concat/values_0'softmax_cross_entropy_with_logits/Slice-softmax_cross_entropy_with_logits/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
║
)softmax_cross_entropy_with_logits/ReshapeReshapeflatten_1/Reshape(softmax_cross_entropy_with_logits/concat*
T0*
Tshape0*0
_output_shapes
:                  
j
(softmax_cross_entropy_with_logits/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
v
)softmax_cross_entropy_with_logits/Shape_2ShapePlaceholder_1*
T0*
_output_shapes
:*
out_type0
k
)softmax_cross_entropy_with_logits/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :
д
'softmax_cross_entropy_with_logits/Sub_1Sub(softmax_cross_entropy_with_logits/Rank_2)softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
Ъ
/softmax_cross_entropy_with_logits/Slice_1/beginPack'softmax_cross_entropy_with_logits/Sub_1*
N*
T0*
_output_shapes
:*

axis 
x
.softmax_cross_entropy_with_logits/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
Ё
)softmax_cross_entropy_with_logits/Slice_1Slice)softmax_cross_entropy_with_logits/Shape_2/softmax_cross_entropy_with_logits/Slice_1/begin.softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:
Ж
3softmax_cross_entropy_with_logits/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
         
q
/softmax_cross_entropy_with_logits/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Б
*softmax_cross_entropy_with_logits/concat_1ConcatV23softmax_cross_entropy_with_logits/concat_1/values_0)softmax_cross_entropy_with_logits/Slice_1/softmax_cross_entropy_with_logits/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
║
+softmax_cross_entropy_with_logits/Reshape_1ReshapePlaceholder_1*softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0*0
_output_shapes
:                  
ф
!softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits)softmax_cross_entropy_with_logits/Reshape+softmax_cross_entropy_with_logits/Reshape_1*
T0*?
_output_shapes-
+:         :                  
k
)softmax_cross_entropy_with_logits/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
в
'softmax_cross_entropy_with_logits/Sub_2Sub&softmax_cross_entropy_with_logits/Rank)softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 
y
/softmax_cross_entropy_with_logits/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: 
Щ
.softmax_cross_entropy_with_logits/Slice_2/sizePack'softmax_cross_entropy_with_logits/Sub_2*
N*
T0*
_output_shapes
:*

axis 
ю
)softmax_cross_entropy_with_logits/Slice_2Slice'softmax_cross_entropy_with_logits/Shape/softmax_cross_entropy_with_logits/Slice_2/begin.softmax_cross_entropy_with_logits/Slice_2/size*
Index0*
T0*
_output_shapes
:
└
+softmax_cross_entropy_with_logits/Reshape_2Reshape!softmax_cross_entropy_with_logits)softmax_cross_entropy_with_logits/Slice_2*
T0*
Tshape0*#
_output_shapes
:         
O
ConstConst*
_output_shapes
:*
dtype0*
valueB: 
~
MeanMean+softmax_cross_entropy_with_logits/Reshape_2Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
S
Rank/packedPackMean*
N*
T0*
_output_shapes
:*

axis 
F
RankConst*
_output_shapes
: *
dtype0*
value	B :
M
range/startConst*
_output_shapes
: *
dtype0*
value	B : 
M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
V
rangeRangerange/startRankrange/delta*

Tidx0*
_output_shapes
:
Q
	Sum/inputPackMean*
N*
T0*
_output_shapes
:*

axis 
Z
SumSum	Sum/inputrange*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
W
SoftmaxSoftmaxflatten_1/Reshape*
T0*'
_output_shapes
:         

j
(softmax_cross_entropy_with_logits_1/RankConst*
_output_shapes
: *
dtype0*
value	B :
|
)softmax_cross_entropy_with_logits_1/ShapeShapeflatten_1/Reshape_1*
T0*
_output_shapes
:*
out_type0
l
*softmax_cross_entropy_with_logits_1/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
~
+softmax_cross_entropy_with_logits_1/Shape_1Shapeflatten_1/Reshape_1*
T0*
_output_shapes
:*
out_type0
k
)softmax_cross_entropy_with_logits_1/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
ж
'softmax_cross_entropy_with_logits_1/SubSub*softmax_cross_entropy_with_logits_1/Rank_1)softmax_cross_entropy_with_logits_1/Sub/y*
T0*
_output_shapes
: 
Ъ
/softmax_cross_entropy_with_logits_1/Slice/beginPack'softmax_cross_entropy_with_logits_1/Sub*
N*
T0*
_output_shapes
:*

axis 
x
.softmax_cross_entropy_with_logits_1/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
Є
)softmax_cross_entropy_with_logits_1/SliceSlice+softmax_cross_entropy_with_logits_1/Shape_1/softmax_cross_entropy_with_logits_1/Slice/begin.softmax_cross_entropy_with_logits_1/Slice/size*
Index0*
T0*
_output_shapes
:
Ж
3softmax_cross_entropy_with_logits_1/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
         
q
/softmax_cross_entropy_with_logits_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Б
*softmax_cross_entropy_with_logits_1/concatConcatV23softmax_cross_entropy_with_logits_1/concat/values_0)softmax_cross_entropy_with_logits_1/Slice/softmax_cross_entropy_with_logits_1/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
└
+softmax_cross_entropy_with_logits_1/ReshapeReshapeflatten_1/Reshape_1*softmax_cross_entropy_with_logits_1/concat*
T0*
Tshape0*0
_output_shapes
:                  
l
*softmax_cross_entropy_with_logits_1/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
r
+softmax_cross_entropy_with_logits_1/Shape_2ShapeSoftmax*
T0*
_output_shapes
:*
out_type0
m
+softmax_cross_entropy_with_logits_1/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :
к
)softmax_cross_entropy_with_logits_1/Sub_1Sub*softmax_cross_entropy_with_logits_1/Rank_2+softmax_cross_entropy_with_logits_1/Sub_1/y*
T0*
_output_shapes
: 
Ю
1softmax_cross_entropy_with_logits_1/Slice_1/beginPack)softmax_cross_entropy_with_logits_1/Sub_1*
N*
T0*
_output_shapes
:*

axis 
z
0softmax_cross_entropy_with_logits_1/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
°
+softmax_cross_entropy_with_logits_1/Slice_1Slice+softmax_cross_entropy_with_logits_1/Shape_21softmax_cross_entropy_with_logits_1/Slice_1/begin0softmax_cross_entropy_with_logits_1/Slice_1/size*
Index0*
T0*
_output_shapes
:
И
5softmax_cross_entropy_with_logits_1/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
         
s
1softmax_cross_entropy_with_logits_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Й
,softmax_cross_entropy_with_logits_1/concat_1ConcatV25softmax_cross_entropy_with_logits_1/concat_1/values_0+softmax_cross_entropy_with_logits_1/Slice_11softmax_cross_entropy_with_logits_1/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
╕
-softmax_cross_entropy_with_logits_1/Reshape_1ReshapeSoftmax,softmax_cross_entropy_with_logits_1/concat_1*
T0*
Tshape0*0
_output_shapes
:                  
ъ
#softmax_cross_entropy_with_logits_1SoftmaxCrossEntropyWithLogits+softmax_cross_entropy_with_logits_1/Reshape-softmax_cross_entropy_with_logits_1/Reshape_1*
T0*?
_output_shapes-
+:         :                  
m
+softmax_cross_entropy_with_logits_1/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
и
)softmax_cross_entropy_with_logits_1/Sub_2Sub(softmax_cross_entropy_with_logits_1/Rank+softmax_cross_entropy_with_logits_1/Sub_2/y*
T0*
_output_shapes
: 
{
1softmax_cross_entropy_with_logits_1/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: 
Э
0softmax_cross_entropy_with_logits_1/Slice_2/sizePack)softmax_cross_entropy_with_logits_1/Sub_2*
N*
T0*
_output_shapes
:*

axis 
Ў
+softmax_cross_entropy_with_logits_1/Slice_2Slice)softmax_cross_entropy_with_logits_1/Shape1softmax_cross_entropy_with_logits_1/Slice_2/begin0softmax_cross_entropy_with_logits_1/Slice_2/size*
Index0*
T0*
_output_shapes
:
╞
-softmax_cross_entropy_with_logits_1/Reshape_2Reshape#softmax_cross_entropy_with_logits_1+softmax_cross_entropy_with_logits_1/Slice_2*
T0*
Tshape0*#
_output_shapes
:         
Q
Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Д
Mean_1Mean-softmax_cross_entropy_with_logits_1/Reshape_2Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
W
Rank_1/packedPackMean_1*
N*
T0*
_output_shapes
:*

axis 
H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
^
range_1Rangerange_1/startRank_1range_1/delta*

Tidx0*
_output_shapes
:
U
Sum_1/inputPackMean_1*
N*
T0*
_output_shapes
:*

axis 
`
Sum_1SumSum_1/inputrange_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
I
AbsAbsconv2d/mul_2*
T0*&
_output_shapes
:
J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
I
addAddV2Absadd/y*
T0*&
_output_shapes
:
@
LogLogadd*
T0*&
_output_shapes
:
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  @
6
Log_1LogConst_2*
T0*
_output_shapes
: 
O
truedivRealDivLogLog_1*
T0*&
_output_shapes
:
L
ReadVariableOpReadVariableOpslope*
_output_shapes
: *
dtype0
T
mulMulReadVariableOptruediv*
T0*&
_output_shapes
:
R
ReadVariableOp_1ReadVariableOp	intercept*
_output_shapes
: *
dtype0
V
add_1AddV2ReadVariableOp_1mul*
T0*&
_output_shapes
:
v
differentiable_roundRoundadd_1*
T0*
_gradient_op_type
Identity*&
_output_shapes
:
`
Const_3Const*
_output_shapes
:*
dtype0*%
valueB"             
g
MinMindifferentiable_roundConst_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
T
Rank_2/packedPackMin*
N*
T0*
_output_shapes
:*

axis 
H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
O
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 
O
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :
^
range_2Rangerange_2/startRank_2range_2/delta*

Tidx0*
_output_shapes
:
R
Min_1/inputPackMin*
N*
T0*
_output_shapes
:*

axis 
`
Min_1MinMin_1/inputrange_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
K
Abs_1Absconv2d/mul_2*
T0*&
_output_shapes
:
L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
O
add_2AddV2Abs_1add_2/y*
T0*&
_output_shapes
:
D
Log_2Logadd_2*
T0*&
_output_shapes
:
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *  @
6
Log_3LogConst_4*
T0*
_output_shapes
: 
S
	truediv_1RealDivLog_2Log_3*
T0*&
_output_shapes
:
N
ReadVariableOp_2ReadVariableOpslope*
_output_shapes
: *
dtype0
Z
mul_1MulReadVariableOp_2	truediv_1*
T0*&
_output_shapes
:
R
ReadVariableOp_3ReadVariableOp	intercept*
_output_shapes
: *
dtype0
X
add_3AddV2ReadVariableOp_3mul_1*
T0*&
_output_shapes
:
x
differentiable_round_1Roundadd_3*
T0*
_gradient_op_type
Identity*&
_output_shapes
:
`
Const_5Const*
_output_shapes
:*
dtype0*%
valueB"             
i
MaxMaxdifferentiable_round_1Const_5*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
T
Rank_3/packedPackMax*
N*
T0*
_output_shapes
:*

axis 
H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :
O
range_3/startConst*
_output_shapes
: *
dtype0*
value	B : 
O
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :
^
range_3Rangerange_3/startRank_3range_3/delta*

Tidx0*
_output_shapes
:
R
Max_1/inputPackMax*
N*
T0*
_output_shapes
:*

axis 
`
Max_1MaxMax_1/inputrange_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
9
subSubMax_1Min_1*
T0*
_output_shapes
: 
L
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
=
add_4AddV2subadd_4/y*
T0*
_output_shapes
: 
4
Abs_2Absadd_4*
T0*
_output_shapes
: 
L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
?
add_5AddV2Abs_2add_5/y*
T0*
_output_shapes
: 
4
Log_4Logadd_5*
T0*
_output_shapes
: 
L
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *  @
6
Log_5LogConst_6*
T0*
_output_shapes
: 
C
	truediv_2RealDivLog_4Log_5*
T0*
_output_shapes
: 
h
differentiable_ceilCeil	truediv_2*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
L
add_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
M
add_6AddV2add_6/xdifferentiable_ceil*
T0*
_output_shapes
: 
M
Abs_3Absconv2d_1/mul_2*
T0*&
_output_shapes
:

L
add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
O
add_7AddV2Abs_3add_7/y*
T0*&
_output_shapes
:

D
Log_6Logadd_7*
T0*&
_output_shapes
:

L
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *  @
6
Log_7LogConst_7*
T0*
_output_shapes
: 
S
	truediv_3RealDivLog_6Log_7*
T0*&
_output_shapes
:

P
ReadVariableOp_4ReadVariableOpslope_1*
_output_shapes
: *
dtype0
Z
mul_2MulReadVariableOp_4	truediv_3*
T0*&
_output_shapes
:

T
ReadVariableOp_5ReadVariableOpintercept_1*
_output_shapes
: *
dtype0
X
add_8AddV2ReadVariableOp_5mul_2*
T0*&
_output_shapes
:

x
differentiable_round_2Roundadd_8*
T0*
_gradient_op_type
Identity*&
_output_shapes
:

`
Const_8Const*
_output_shapes
:*
dtype0*%
valueB"             
k
Min_2Mindifferentiable_round_2Const_8*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
V
Rank_4/packedPackMin_2*
N*
T0*
_output_shapes
:*

axis 
H
Rank_4Const*
_output_shapes
: *
dtype0*
value	B :
O
range_4/startConst*
_output_shapes
: *
dtype0*
value	B : 
O
range_4/deltaConst*
_output_shapes
: *
dtype0*
value	B :
^
range_4Rangerange_4/startRank_4range_4/delta*

Tidx0*
_output_shapes
:
T
Min_3/inputPackMin_2*
N*
T0*
_output_shapes
:*

axis 
`
Min_3MinMin_3/inputrange_4*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
M
Abs_4Absconv2d_1/mul_2*
T0*&
_output_shapes
:

L
add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
O
add_9AddV2Abs_4add_9/y*
T0*&
_output_shapes
:

D
Log_8Logadd_9*
T0*&
_output_shapes
:

L
Const_9Const*
_output_shapes
: *
dtype0*
valueB
 *  @
6
Log_9LogConst_9*
T0*
_output_shapes
: 
S
	truediv_4RealDivLog_8Log_9*
T0*&
_output_shapes
:

P
ReadVariableOp_6ReadVariableOpslope_1*
_output_shapes
: *
dtype0
Z
mul_3MulReadVariableOp_6	truediv_4*
T0*&
_output_shapes
:

T
ReadVariableOp_7ReadVariableOpintercept_1*
_output_shapes
: *
dtype0
Y
add_10AddV2ReadVariableOp_7mul_3*
T0*&
_output_shapes
:

y
differentiable_round_3Roundadd_10*
T0*
_gradient_op_type
Identity*&
_output_shapes
:

a
Const_10Const*
_output_shapes
:*
dtype0*%
valueB"             
l
Max_2Maxdifferentiable_round_3Const_10*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
V
Rank_5/packedPackMax_2*
N*
T0*
_output_shapes
:*

axis 
H
Rank_5Const*
_output_shapes
: *
dtype0*
value	B :
O
range_5/startConst*
_output_shapes
: *
dtype0*
value	B : 
O
range_5/deltaConst*
_output_shapes
: *
dtype0*
value	B :
^
range_5Rangerange_5/startRank_5range_5/delta*

Tidx0*
_output_shapes
:
T
Max_3/inputPackMax_2*
N*
T0*
_output_shapes
:*

axis 
`
Max_3MaxMax_3/inputrange_5*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
;
sub_1SubMax_3Min_3*
T0*
_output_shapes
: 
M
add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
A
add_11AddV2sub_1add_11/y*
T0*
_output_shapes
: 
5
Abs_5Absadd_11*
T0*
_output_shapes
: 
M
add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
A
add_12AddV2Abs_5add_12/y*
T0*
_output_shapes
: 
6
Log_10Logadd_12*
T0*
_output_shapes
: 
M
Const_11Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_11LogConst_11*
T0*
_output_shapes
: 
E
	truediv_5RealDivLog_10Log_11*
T0*
_output_shapes
: 
j
differentiable_ceil_1Ceil	truediv_5*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
M
add_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Q
add_13AddV2add_13/xdifferentiable_ceil_1*
T0*
_output_shapes
: 
N
Abs_6Absconv2d_2/mul_2*
T0*'
_output_shapes
:1
А
M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
R
add_14AddV2Abs_6add_14/y*
T0*'
_output_shapes
:1
А
G
Log_12Logadd_14*
T0*'
_output_shapes
:1
А
M
Const_12Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_13LogConst_12*
T0*
_output_shapes
: 
V
	truediv_6RealDivLog_12Log_13*
T0*'
_output_shapes
:1
А
P
ReadVariableOp_8ReadVariableOpslope_2*
_output_shapes
: *
dtype0
[
mul_4MulReadVariableOp_8	truediv_6*
T0*'
_output_shapes
:1
А
T
ReadVariableOp_9ReadVariableOpintercept_2*
_output_shapes
: *
dtype0
Z
add_15AddV2ReadVariableOp_9mul_4*
T0*'
_output_shapes
:1
А
z
differentiable_round_4Roundadd_15*
T0*
_gradient_op_type
Identity*'
_output_shapes
:1
А
a
Const_13Const*
_output_shapes
:*
dtype0*%
valueB"             
l
Min_4Mindifferentiable_round_4Const_13*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
V
Rank_6/packedPackMin_4*
N*
T0*
_output_shapes
:*

axis 
H
Rank_6Const*
_output_shapes
: *
dtype0*
value	B :
O
range_6/startConst*
_output_shapes
: *
dtype0*
value	B : 
O
range_6/deltaConst*
_output_shapes
: *
dtype0*
value	B :
^
range_6Rangerange_6/startRank_6range_6/delta*

Tidx0*
_output_shapes
:
T
Min_5/inputPackMin_4*
N*
T0*
_output_shapes
:*

axis 
`
Min_5MinMin_5/inputrange_6*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
N
Abs_7Absconv2d_2/mul_2*
T0*'
_output_shapes
:1
А
M
add_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
R
add_16AddV2Abs_7add_16/y*
T0*'
_output_shapes
:1
А
G
Log_14Logadd_16*
T0*'
_output_shapes
:1
А
M
Const_14Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_15LogConst_14*
T0*
_output_shapes
: 
V
	truediv_7RealDivLog_14Log_15*
T0*'
_output_shapes
:1
А
Q
ReadVariableOp_10ReadVariableOpslope_2*
_output_shapes
: *
dtype0
\
mul_5MulReadVariableOp_10	truediv_7*
T0*'
_output_shapes
:1
А
U
ReadVariableOp_11ReadVariableOpintercept_2*
_output_shapes
: *
dtype0
[
add_17AddV2ReadVariableOp_11mul_5*
T0*'
_output_shapes
:1
А
z
differentiable_round_5Roundadd_17*
T0*
_gradient_op_type
Identity*'
_output_shapes
:1
А
a
Const_15Const*
_output_shapes
:*
dtype0*%
valueB"             
l
Max_4Maxdifferentiable_round_5Const_15*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
V
Rank_7/packedPackMax_4*
N*
T0*
_output_shapes
:*

axis 
H
Rank_7Const*
_output_shapes
: *
dtype0*
value	B :
O
range_7/startConst*
_output_shapes
: *
dtype0*
value	B : 
O
range_7/deltaConst*
_output_shapes
: *
dtype0*
value	B :
^
range_7Rangerange_7/startRank_7range_7/delta*

Tidx0*
_output_shapes
:
T
Max_5/inputPackMax_4*
N*
T0*
_output_shapes
:*

axis 
`
Max_5MaxMax_5/inputrange_7*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
;
sub_2SubMax_5Min_5*
T0*
_output_shapes
: 
M
add_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
A
add_18AddV2sub_2add_18/y*
T0*
_output_shapes
: 
5
Abs_8Absadd_18*
T0*
_output_shapes
: 
M
add_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
A
add_19AddV2Abs_8add_19/y*
T0*
_output_shapes
: 
6
Log_16Logadd_19*
T0*
_output_shapes
: 
M
Const_16Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_17LogConst_16*
T0*
_output_shapes
: 
E
	truediv_8RealDivLog_16Log_17*
T0*
_output_shapes
: 
j
differentiable_ceil_2Ceil	truediv_8*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
M
add_20/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Q
add_20AddV2add_20/xdifferentiable_ceil_2*
T0*
_output_shapes
: 
X
Abs_9Absconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
M
add_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
[
add_21AddV2Abs_9add_21/y*
T0*0
_output_shapes
:         А
P
Log_18Logadd_21*
T0*0
_output_shapes
:         А
M
Const_17Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_19LogConst_17*
T0*
_output_shapes
: 
_
	truediv_9RealDivLog_18Log_19*
T0*0
_output_shapes
:         А
Q
ReadVariableOp_12ReadVariableOpslope_3*
_output_shapes
: *
dtype0
e
mul_6MulReadVariableOp_12	truediv_9*
T0*0
_output_shapes
:         А
U
ReadVariableOp_13ReadVariableOpintercept_3*
_output_shapes
: *
dtype0
d
add_22AddV2ReadVariableOp_13mul_6*
T0*0
_output_shapes
:         А
Г
differentiable_round_6Roundadd_22*
T0*
_gradient_op_type
Identity*0
_output_shapes
:         А
a
Const_18Const*
_output_shapes
:*
dtype0*%
valueB"             
l
Min_6Mindifferentiable_round_6Const_18*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
V
Rank_8/packedPackMin_6*
N*
T0*
_output_shapes
:*

axis 
H
Rank_8Const*
_output_shapes
: *
dtype0*
value	B :
O
range_8/startConst*
_output_shapes
: *
dtype0*
value	B : 
O
range_8/deltaConst*
_output_shapes
: *
dtype0*
value	B :
^
range_8Rangerange_8/startRank_8range_8/delta*

Tidx0*
_output_shapes
:
T
Min_7/inputPackMin_6*
N*
T0*
_output_shapes
:*

axis 
`
Min_7MinMin_7/inputrange_8*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Y
Abs_10Absconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
M
add_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
\
add_23AddV2Abs_10add_23/y*
T0*0
_output_shapes
:         А
P
Log_20Logadd_23*
T0*0
_output_shapes
:         А
M
Const_19Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_21LogConst_19*
T0*
_output_shapes
: 
`

truediv_10RealDivLog_20Log_21*
T0*0
_output_shapes
:         А
Q
ReadVariableOp_14ReadVariableOpslope_3*
_output_shapes
: *
dtype0
f
mul_7MulReadVariableOp_14
truediv_10*
T0*0
_output_shapes
:         А
U
ReadVariableOp_15ReadVariableOpintercept_3*
_output_shapes
: *
dtype0
d
add_24AddV2ReadVariableOp_15mul_7*
T0*0
_output_shapes
:         А
Г
differentiable_round_7Roundadd_24*
T0*
_gradient_op_type
Identity*0
_output_shapes
:         А
a
Const_20Const*
_output_shapes
:*
dtype0*%
valueB"             
l
Max_6Maxdifferentiable_round_7Const_20*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
V
Rank_9/packedPackMax_6*
N*
T0*
_output_shapes
:*

axis 
H
Rank_9Const*
_output_shapes
: *
dtype0*
value	B :
O
range_9/startConst*
_output_shapes
: *
dtype0*
value	B : 
O
range_9/deltaConst*
_output_shapes
: *
dtype0*
value	B :
^
range_9Rangerange_9/startRank_9range_9/delta*

Tidx0*
_output_shapes
:
T
Max_7/inputPackMax_6*
N*
T0*
_output_shapes
:*

axis 
`
Max_7MaxMax_7/inputrange_9*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
;
sub_3SubMax_7Min_7*
T0*
_output_shapes
: 
M
add_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
A
add_25AddV2sub_3add_25/y*
T0*
_output_shapes
: 
6
Abs_11Absadd_25*
T0*
_output_shapes
: 
M
add_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
B
add_26AddV2Abs_11add_26/y*
T0*
_output_shapes
: 
6
Log_22Logadd_26*
T0*
_output_shapes
: 
M
Const_21Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_23LogConst_21*
T0*
_output_shapes
: 
F

truediv_11RealDivLog_22Log_23*
T0*
_output_shapes
: 
k
differentiable_ceil_3Ceil
truediv_11*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
M
add_27/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Q
add_27AddV2add_27/xdifferentiable_ceil_3*
T0*
_output_shapes
: 
P
Abs_12Absconv2d_3/mul_2*
T0*(
_output_shapes
:АА
M
add_28/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_28AddV2Abs_12add_28/y*
T0*(
_output_shapes
:АА
H
Log_24Logadd_28*
T0*(
_output_shapes
:АА
M
Const_22Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_25LogConst_22*
T0*
_output_shapes
: 
X

truediv_12RealDivLog_24Log_25*
T0*(
_output_shapes
:АА
Q
ReadVariableOp_16ReadVariableOpslope_4*
_output_shapes
: *
dtype0
^
mul_8MulReadVariableOp_16
truediv_12*
T0*(
_output_shapes
:АА
U
ReadVariableOp_17ReadVariableOpintercept_4*
_output_shapes
: *
dtype0
\
add_29AddV2ReadVariableOp_17mul_8*
T0*(
_output_shapes
:АА
{
differentiable_round_8Roundadd_29*
T0*
_gradient_op_type
Identity*(
_output_shapes
:АА
a
Const_23Const*
_output_shapes
:*
dtype0*%
valueB"             
l
Min_8Mindifferentiable_round_8Const_23*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
W
Rank_10/packedPackMin_8*
N*
T0*
_output_shapes
:*

axis 
I
Rank_10Const*
_output_shapes
: *
dtype0*
value	B :
P
range_10/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_10/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_10Rangerange_10/startRank_10range_10/delta*

Tidx0*
_output_shapes
:
T
Min_9/inputPackMin_8*
N*
T0*
_output_shapes
:*

axis 
a
Min_9MinMin_9/inputrange_10*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
P
Abs_13Absconv2d_3/mul_2*
T0*(
_output_shapes
:АА
M
add_30/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_30AddV2Abs_13add_30/y*
T0*(
_output_shapes
:АА
H
Log_26Logadd_30*
T0*(
_output_shapes
:АА
M
Const_24Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_27LogConst_24*
T0*
_output_shapes
: 
X

truediv_13RealDivLog_26Log_27*
T0*(
_output_shapes
:АА
Q
ReadVariableOp_18ReadVariableOpslope_4*
_output_shapes
: *
dtype0
^
mul_9MulReadVariableOp_18
truediv_13*
T0*(
_output_shapes
:АА
U
ReadVariableOp_19ReadVariableOpintercept_4*
_output_shapes
: *
dtype0
\
add_31AddV2ReadVariableOp_19mul_9*
T0*(
_output_shapes
:АА
{
differentiable_round_9Roundadd_31*
T0*
_gradient_op_type
Identity*(
_output_shapes
:АА
a
Const_25Const*
_output_shapes
:*
dtype0*%
valueB"             
l
Max_8Maxdifferentiable_round_9Const_25*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
W
Rank_11/packedPackMax_8*
N*
T0*
_output_shapes
:*

axis 
I
Rank_11Const*
_output_shapes
: *
dtype0*
value	B :
P
range_11/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_11/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_11Rangerange_11/startRank_11range_11/delta*

Tidx0*
_output_shapes
:
T
Max_9/inputPackMax_8*
N*
T0*
_output_shapes
:*

axis 
a
Max_9MaxMax_9/inputrange_11*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
;
sub_4SubMax_9Min_9*
T0*
_output_shapes
: 
M
add_32/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
A
add_32AddV2sub_4add_32/y*
T0*
_output_shapes
: 
6
Abs_14Absadd_32*
T0*
_output_shapes
: 
M
add_33/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
B
add_33AddV2Abs_14add_33/y*
T0*
_output_shapes
: 
6
Log_28Logadd_33*
T0*
_output_shapes
: 
M
Const_26Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_29LogConst_26*
T0*
_output_shapes
: 
F

truediv_14RealDivLog_28Log_29*
T0*
_output_shapes
: 
k
differentiable_ceil_4Ceil
truediv_14*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
M
add_34/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Q
add_34AddV2add_34/xdifferentiable_ceil_4*
T0*
_output_shapes
: 
O
Abs_15Absconv2d_4/mul_2*
T0*'
_output_shapes
:А

M
add_35/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
S
add_35AddV2Abs_15add_35/y*
T0*'
_output_shapes
:А

G
Log_30Logadd_35*
T0*'
_output_shapes
:А

M
Const_27Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_31LogConst_27*
T0*
_output_shapes
: 
W

truediv_15RealDivLog_30Log_31*
T0*'
_output_shapes
:А

Q
ReadVariableOp_20ReadVariableOpslope_5*
_output_shapes
: *
dtype0
^
mul_10MulReadVariableOp_20
truediv_15*
T0*'
_output_shapes
:А

U
ReadVariableOp_21ReadVariableOpintercept_5*
_output_shapes
: *
dtype0
\
add_36AddV2ReadVariableOp_21mul_10*
T0*'
_output_shapes
:А

{
differentiable_round_10Roundadd_36*
T0*
_gradient_op_type
Identity*'
_output_shapes
:А

a
Const_28Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Min_10Mindifferentiable_round_10Const_28*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_12/packedPackMin_10*
N*
T0*
_output_shapes
:*

axis 
I
Rank_12Const*
_output_shapes
: *
dtype0*
value	B :
P
range_12/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_12/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_12Rangerange_12/startRank_12range_12/delta*

Tidx0*
_output_shapes
:
V
Min_11/inputPackMin_10*
N*
T0*
_output_shapes
:*

axis 
c
Min_11MinMin_11/inputrange_12*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
O
Abs_16Absconv2d_4/mul_2*
T0*'
_output_shapes
:А

M
add_37/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
S
add_37AddV2Abs_16add_37/y*
T0*'
_output_shapes
:А

G
Log_32Logadd_37*
T0*'
_output_shapes
:А

M
Const_29Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_33LogConst_29*
T0*
_output_shapes
: 
W

truediv_16RealDivLog_32Log_33*
T0*'
_output_shapes
:А

Q
ReadVariableOp_22ReadVariableOpslope_5*
_output_shapes
: *
dtype0
^
mul_11MulReadVariableOp_22
truediv_16*
T0*'
_output_shapes
:А

U
ReadVariableOp_23ReadVariableOpintercept_5*
_output_shapes
: *
dtype0
\
add_38AddV2ReadVariableOp_23mul_11*
T0*'
_output_shapes
:А

{
differentiable_round_11Roundadd_38*
T0*
_gradient_op_type
Identity*'
_output_shapes
:А

a
Const_30Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Max_10Maxdifferentiable_round_11Const_30*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_13/packedPackMax_10*
N*
T0*
_output_shapes
:*

axis 
I
Rank_13Const*
_output_shapes
: *
dtype0*
value	B :
P
range_13/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_13/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_13Rangerange_13/startRank_13range_13/delta*

Tidx0*
_output_shapes
:
V
Max_11/inputPackMax_10*
N*
T0*
_output_shapes
:*

axis 
c
Max_11MaxMax_11/inputrange_13*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
=
sub_5SubMax_11Min_11*
T0*
_output_shapes
: 
M
add_39/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
A
add_39AddV2sub_5add_39/y*
T0*
_output_shapes
: 
6
Abs_17Absadd_39*
T0*
_output_shapes
: 
M
add_40/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
B
add_40AddV2Abs_17add_40/y*
T0*
_output_shapes
: 
6
Log_34Logadd_40*
T0*
_output_shapes
: 
M
Const_31Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_35LogConst_31*
T0*
_output_shapes
: 
F

truediv_17RealDivLog_34Log_35*
T0*
_output_shapes
: 
k
differentiable_ceil_5Ceil
truediv_17*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
M
add_41/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Q
add_41AddV2add_41/xdifferentiable_ceil_5*
T0*
_output_shapes
: 
J
pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
9
powPowpow/xadd_6*
T0*
_output_shapes
: 
L
pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
>
pow_1Powpow_1/xadd_13*
T0*
_output_shapes
: 
L
pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
>
pow_2Powpow_2/xadd_20*
T0*
_output_shapes
: 
L
pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
>
pow_3Powpow_3/xadd_27*
T0*
_output_shapes
: 
L
pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
>
pow_4Powpow_4/xadd_34*
T0*
_output_shapes
: 
L
pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
>
pow_5Powpow_5/xadd_41*
T0*
_output_shapes
: 
x
Rank_14/packedPackpowpow_1pow_2pow_3pow_4pow_5*
N*
T0*
_output_shapes
:*

axis 
I
Rank_14Const*
_output_shapes
: *
dtype0*
value	B :
P
range_14/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_14/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_14Rangerange_14/startRank_14range_14/delta*

Tidx0*
_output_shapes
:
u
Sum_2/inputPackpowpow_1pow_2pow_3pow_4pow_5*
N*
T0*
_output_shapes
:*

axis 
a
Sum_2SumSum_2/inputrange_14*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
k
L2Loss/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
H
L2LossL2LossL2Loss/ReadVariableOp*
T0*
_output_shapes
: 
M
add_42/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
B
add_42AddV2add_42/xL2Loss*
T0*
_output_shapes
: 
o
L2Loss_1/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
L
L2Loss_1L2LossL2Loss_1/ReadVariableOp*
T0*
_output_shapes
: 
B
add_43AddV2add_42L2Loss_1*
T0*
_output_shapes
: 
p
L2Loss_2/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
L
L2Loss_2L2LossL2Loss_2/ReadVariableOp*
T0*
_output_shapes
: 
B
add_44AddV2add_43L2Loss_2*
T0*
_output_shapes
: 
q
L2Loss_3/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
L
L2Loss_3L2LossL2Loss_3/ReadVariableOp*
T0*
_output_shapes
: 
B
add_45AddV2add_44L2Loss_3*
T0*
_output_shapes
: 
p
L2Loss_4/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
L
L2Loss_4L2LossL2Loss_4/ReadVariableOp*
T0*
_output_shapes
: 
B
add_46AddV2add_45L2Loss_4*
T0*
_output_shapes
: 
m
L2Loss_5/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
L
L2Loss_5L2LossL2Loss_5/ReadVariableOp*
T0*
_output_shapes
: 
M
add_47/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
D
add_47AddV2add_47/xL2Loss_5*
T0*
_output_shapes
: 
o
L2Loss_6/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
L
L2Loss_6L2LossL2Loss_6/ReadVariableOp*
T0*
_output_shapes
: 
B
add_48AddV2add_47L2Loss_6*
T0*
_output_shapes
: 
p
L2Loss_7/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
L
L2Loss_7L2LossL2Loss_7/ReadVariableOp*
T0*
_output_shapes
: 
B
add_49AddV2add_48L2Loss_7*
T0*
_output_shapes
: 
q
L2Loss_8/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
L
L2Loss_8L2LossL2Loss_8/ReadVariableOp*
T0*
_output_shapes
: 
B
add_50AddV2add_49L2Loss_8*
T0*
_output_shapes
: 
p
L2Loss_9/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
L
L2Loss_9L2LossL2Loss_9/ReadVariableOp*
T0*
_output_shapes
: 
B
add_51AddV2add_50L2Loss_9*
T0*
_output_shapes
: 
M
Const_32Const*
_output_shapes
: *
dtype0*
valueB
 *
╫#<
M
Const_33Const*
_output_shapes
: *
dtype0*
valueB
 *м┼'7
M
Const_34Const*
_output_shapes
: *
dtype0*
valueB
 *╖╤8
?
mul_12MulSum_1Const_32*
T0*
_output_shapes
: 
=
add_52AddV2Summul_12*
T0*
_output_shapes
: 
?
mul_13MulSum_2Const_33*
T0*
_output_shapes
: 
@
add_53AddV2add_52mul_13*
T0*
_output_shapes
: 
@
mul_14MulConst_34add_46*
T0*
_output_shapes
: 
@
add_54AddV2add_53mul_14*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
X
gradients/grad_ys_0Const*
_output_shapes
: *
dtype0*
valueB
 *  А?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*
_output_shapes
: *

index_type0
?
&gradients/add_54_grad/tuple/group_depsNoOp^gradients/Fill
╖
.gradients/add_54_grad/tuple/control_dependencyIdentitygradients/Fill'^gradients/add_54_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
╣
0gradients/add_54_grad/tuple/control_dependency_1Identitygradients/Fill'^gradients/add_54_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
_
&gradients/add_53_grad/tuple/group_depsNoOp/^gradients/add_54_grad/tuple/control_dependency
╫
.gradients/add_53_grad/tuple/control_dependencyIdentity.gradients/add_54_grad/tuple/control_dependency'^gradients/add_53_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
┘
0gradients/add_53_grad/tuple/control_dependency_1Identity.gradients/add_54_grad/tuple/control_dependency'^gradients/add_53_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
{
gradients/mul_14_grad/MulMul0gradients/add_54_grad/tuple/control_dependency_1add_46*
T0*
_output_shapes
: 

gradients/mul_14_grad/Mul_1Mul0gradients/add_54_grad/tuple/control_dependency_1Const_34*
T0*
_output_shapes
: 
h
&gradients/mul_14_grad/tuple/group_depsNoOp^gradients/mul_14_grad/Mul^gradients/mul_14_grad/Mul_1
═
.gradients/mul_14_grad/tuple/control_dependencyIdentitygradients/mul_14_grad/Mul'^gradients/mul_14_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/mul_14_grad/Mul*
_output_shapes
: 
╙
0gradients/mul_14_grad/tuple/control_dependency_1Identitygradients/mul_14_grad/Mul_1'^gradients/mul_14_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_14_grad/Mul_1*
_output_shapes
: 
_
&gradients/add_52_grad/tuple/group_depsNoOp/^gradients/add_53_grad/tuple/control_dependency
╫
.gradients/add_52_grad/tuple/control_dependencyIdentity.gradients/add_53_grad/tuple/control_dependency'^gradients/add_52_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
┘
0gradients/add_52_grad/tuple/control_dependency_1Identity.gradients/add_53_grad/tuple/control_dependency'^gradients/add_52_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
}
gradients/mul_13_grad/MulMul0gradients/add_53_grad/tuple/control_dependency_1Const_33*
T0*
_output_shapes
: 
|
gradients/mul_13_grad/Mul_1Mul0gradients/add_53_grad/tuple/control_dependency_1Sum_2*
T0*
_output_shapes
: 
h
&gradients/mul_13_grad/tuple/group_depsNoOp^gradients/mul_13_grad/Mul^gradients/mul_13_grad/Mul_1
═
.gradients/mul_13_grad/tuple/control_dependencyIdentitygradients/mul_13_grad/Mul'^gradients/mul_13_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/mul_13_grad/Mul*
_output_shapes
: 
╙
0gradients/mul_13_grad/tuple/control_dependency_1Identitygradients/mul_13_grad/Mul_1'^gradients/mul_13_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_13_grad/Mul_1*
_output_shapes
: 
a
&gradients/add_46_grad/tuple/group_depsNoOp1^gradients/mul_14_grad/tuple/control_dependency_1
ц
.gradients/add_46_grad/tuple/control_dependencyIdentity0gradients/mul_14_grad/tuple/control_dependency_1'^gradients/add_46_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_14_grad/Mul_1*
_output_shapes
: 
ш
0gradients/add_46_grad/tuple/control_dependency_1Identity0gradients/mul_14_grad/tuple/control_dependency_1'^gradients/add_46_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_14_grad/Mul_1*
_output_shapes
: 
j
 gradients/Sum_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
к
gradients/Sum_grad/ReshapeReshape.gradients/add_52_grad/tuple/control_dependency gradients/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
b
gradients/Sum_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB:
М
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/Const*
T0*

Tmultiples0*
_output_shapes
:
}
gradients/mul_12_grad/MulMul0gradients/add_52_grad/tuple/control_dependency_1Const_32*
T0*
_output_shapes
: 
|
gradients/mul_12_grad/Mul_1Mul0gradients/add_52_grad/tuple/control_dependency_1Sum_1*
T0*
_output_shapes
: 
h
&gradients/mul_12_grad/tuple/group_depsNoOp^gradients/mul_12_grad/Mul^gradients/mul_12_grad/Mul_1
═
.gradients/mul_12_grad/tuple/control_dependencyIdentitygradients/mul_12_grad/Mul'^gradients/mul_12_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/mul_12_grad/Mul*
_output_shapes
: 
╙
0gradients/mul_12_grad/tuple/control_dependency_1Identitygradients/mul_12_grad/Mul_1'^gradients/mul_12_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_12_grad/Mul_1*
_output_shapes
: 
l
"gradients/Sum_2_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
о
gradients/Sum_2_grad/ReshapeReshape.gradients/mul_13_grad/tuple/control_dependency"gradients/Sum_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
d
gradients/Sum_2_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB:
Т
gradients/Sum_2_grad/TileTilegradients/Sum_2_grad/Reshapegradients/Sum_2_grad/Const*
T0*

Tmultiples0*
_output_shapes
:
_
&gradients/add_45_grad/tuple/group_depsNoOp/^gradients/add_46_grad/tuple/control_dependency
ф
.gradients/add_45_grad/tuple/control_dependencyIdentity.gradients/add_46_grad/tuple/control_dependency'^gradients/add_45_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_14_grad/Mul_1*
_output_shapes
: 
ц
0gradients/add_45_grad/tuple/control_dependency_1Identity.gradients/add_46_grad/tuple/control_dependency'^gradients/add_45_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_14_grad/Mul_1*
_output_shapes
: 
Я
gradients/L2Loss_4_grad/mulMulL2Loss_4/ReadVariableOp0gradients/add_46_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:А

{
 gradients/Sum/input_grad/unstackUnpackgradients/Sum_grad/Tile*
T0*
_output_shapes
: *

axis *	
num
l
"gradients/Sum_1_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
о
gradients/Sum_1_grad/ReshapeReshape.gradients/mul_12_grad/tuple/control_dependency"gradients/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
d
gradients/Sum_1_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB:
Т
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/Const*
T0*

Tmultiples0*
_output_shapes
:
Й
"gradients/Sum_2/input_grad/unstackUnpackgradients/Sum_2_grad/Tile*
T0* 
_output_shapes
: : : : : : *

axis *	
num
X
+gradients/Sum_2/input_grad/tuple/group_depsNoOp#^gradients/Sum_2/input_grad/unstack
щ
3gradients/Sum_2/input_grad/tuple/control_dependencyIdentity"gradients/Sum_2/input_grad/unstack,^gradients/Sum_2/input_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Sum_2/input_grad/unstack*
_output_shapes
: 
э
5gradients/Sum_2/input_grad/tuple/control_dependency_1Identity$gradients/Sum_2/input_grad/unstack:1,^gradients/Sum_2/input_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Sum_2/input_grad/unstack*
_output_shapes
: 
э
5gradients/Sum_2/input_grad/tuple/control_dependency_2Identity$gradients/Sum_2/input_grad/unstack:2,^gradients/Sum_2/input_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Sum_2/input_grad/unstack*
_output_shapes
: 
э
5gradients/Sum_2/input_grad/tuple/control_dependency_3Identity$gradients/Sum_2/input_grad/unstack:3,^gradients/Sum_2/input_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Sum_2/input_grad/unstack*
_output_shapes
: 
э
5gradients/Sum_2/input_grad/tuple/control_dependency_4Identity$gradients/Sum_2/input_grad/unstack:4,^gradients/Sum_2/input_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Sum_2/input_grad/unstack*
_output_shapes
: 
э
5gradients/Sum_2/input_grad/tuple/control_dependency_5Identity$gradients/Sum_2/input_grad/unstack:5,^gradients/Sum_2/input_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Sum_2/input_grad/unstack*
_output_shapes
: 
_
&gradients/add_44_grad/tuple/group_depsNoOp/^gradients/add_45_grad/tuple/control_dependency
ф
.gradients/add_44_grad/tuple/control_dependencyIdentity.gradients/add_45_grad/tuple/control_dependency'^gradients/add_44_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_14_grad/Mul_1*
_output_shapes
: 
ц
0gradients/add_44_grad/tuple/control_dependency_1Identity.gradients/add_45_grad/tuple/control_dependency'^gradients/add_44_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_14_grad/Mul_1*
_output_shapes
: 
а
gradients/L2Loss_3_grad/mulMulL2Loss_3/ReadVariableOp0gradients/add_45_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:АА
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Ю
gradients/Mean_grad/ReshapeReshape gradients/Sum/input_grad/unstack!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
Д
gradients/Mean_grad/ShapeShape+softmax_cross_entropy_with_logits/Reshape_2*
T0*
_output_shapes
:*
out_type0
Ш
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:         
Ж
gradients/Mean_grad/Shape_1Shape+softmax_cross_entropy_with_logits/Reshape_2*
T0*
_output_shapes
:*
out_type0
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
c
gradients/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
_
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
А
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
И
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:         

"gradients/Sum_1/input_grad/unstackUnpackgradients/Sum_1_grad/Tile*
T0*
_output_shapes
: *

axis *	
num
n
+gradients/pow_grad/BroadcastGradientArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
n
+gradients/pow_grad/BroadcastGradientArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 
╪
(gradients/pow_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/pow_grad/BroadcastGradientArgs/s0+gradients/pow_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:         :         
z
gradients/pow_grad/mulMul3gradients/Sum_2/input_grad/tuple/control_dependencyadd_6*
T0*
_output_shapes
: 
]
gradients/pow_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
_
gradients/pow_grad/subSubadd_6gradients/pow_grad/sub/y*
T0*
_output_shapes
: 
]
gradients/pow_grad/PowPowpow/xgradients/pow_grad/sub*
T0*
_output_shapes
: 
p
gradients/pow_grad/mul_1Mulgradients/pow_grad/mulgradients/pow_grad/Pow*
T0*
_output_shapes
: 
a
gradients/pow_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
k
gradients/pow_grad/GreaterGreaterpow/xgradients/pow_grad/Greater/y*
T0*
_output_shapes
: 
e
"gradients/pow_grad/ones_like/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
g
"gradients/pow_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Я
gradients/pow_grad/ones_likeFill"gradients/pow_grad/ones_like/Shape"gradients/pow_grad/ones_like/Const*
T0*
_output_shapes
: *

index_type0
Е
gradients/pow_grad/SelectSelectgradients/pow_grad/Greaterpow/xgradients/pow_grad/ones_like*
T0*
_output_shapes
: 
Y
gradients/pow_grad/LogLoggradients/pow_grad/Select*
T0*
_output_shapes
: 
b
gradients/pow_grad/zeros_likeConst*
_output_shapes
: *
dtype0*
valueB
 *    
Щ
gradients/pow_grad/Select_1Selectgradients/pow_grad/Greatergradients/pow_grad/Loggradients/pow_grad/zeros_like*
T0*
_output_shapes
: 
z
gradients/pow_grad/mul_2Mul3gradients/Sum_2/input_grad/tuple/control_dependencypow*
T0*
_output_shapes
: 
w
gradients/pow_grad/mul_3Mulgradients/pow_grad/mul_2gradients/pow_grad/Select_1*
T0*
_output_shapes
: 
a
#gradients/pow_grad/tuple/group_depsNoOp^gradients/pow_grad/mul_1^gradients/pow_grad/mul_3
┼
+gradients/pow_grad/tuple/control_dependencyIdentitygradients/pow_grad/mul_1$^gradients/pow_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/pow_grad/mul_1*
_output_shapes
: 
╟
-gradients/pow_grad/tuple/control_dependency_1Identitygradients/pow_grad/mul_3$^gradients/pow_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/pow_grad/mul_3*
_output_shapes
: 

gradients/pow_1_grad/mulMul5gradients/Sum_2/input_grad/tuple/control_dependency_1add_13*
T0*
_output_shapes
: 
_
gradients/pow_1_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
d
gradients/pow_1_grad/subSubadd_13gradients/pow_1_grad/sub/y*
T0*
_output_shapes
: 
c
gradients/pow_1_grad/PowPowpow_1/xgradients/pow_1_grad/sub*
T0*
_output_shapes
: 
v
gradients/pow_1_grad/mul_1Mulgradients/pow_1_grad/mulgradients/pow_1_grad/Pow*
T0*
_output_shapes
: 
c
gradients/pow_1_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
q
gradients/pow_1_grad/GreaterGreaterpow_1/xgradients/pow_1_grad/Greater/y*
T0*
_output_shapes
: 
g
$gradients/pow_1_grad/ones_like/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
i
$gradients/pow_1_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
е
gradients/pow_1_grad/ones_likeFill$gradients/pow_1_grad/ones_like/Shape$gradients/pow_1_grad/ones_like/Const*
T0*
_output_shapes
: *

index_type0
Н
gradients/pow_1_grad/SelectSelectgradients/pow_1_grad/Greaterpow_1/xgradients/pow_1_grad/ones_like*
T0*
_output_shapes
: 
]
gradients/pow_1_grad/LogLoggradients/pow_1_grad/Select*
T0*
_output_shapes
: 
d
gradients/pow_1_grad/zeros_likeConst*
_output_shapes
: *
dtype0*
valueB
 *    
б
gradients/pow_1_grad/Select_1Selectgradients/pow_1_grad/Greatergradients/pow_1_grad/Loggradients/pow_1_grad/zeros_like*
T0*
_output_shapes
: 
А
gradients/pow_1_grad/mul_2Mul5gradients/Sum_2/input_grad/tuple/control_dependency_1pow_1*
T0*
_output_shapes
: 
}
gradients/pow_1_grad/mul_3Mulgradients/pow_1_grad/mul_2gradients/pow_1_grad/Select_1*
T0*
_output_shapes
: 
g
%gradients/pow_1_grad/tuple/group_depsNoOp^gradients/pow_1_grad/mul_1^gradients/pow_1_grad/mul_3
═
-gradients/pow_1_grad/tuple/control_dependencyIdentitygradients/pow_1_grad/mul_1&^gradients/pow_1_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_1_grad/mul_1*
_output_shapes
: 
╧
/gradients/pow_1_grad/tuple/control_dependency_1Identitygradients/pow_1_grad/mul_3&^gradients/pow_1_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_1_grad/mul_3*
_output_shapes
: 

gradients/pow_2_grad/mulMul5gradients/Sum_2/input_grad/tuple/control_dependency_2add_20*
T0*
_output_shapes
: 
_
gradients/pow_2_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
d
gradients/pow_2_grad/subSubadd_20gradients/pow_2_grad/sub/y*
T0*
_output_shapes
: 
c
gradients/pow_2_grad/PowPowpow_2/xgradients/pow_2_grad/sub*
T0*
_output_shapes
: 
v
gradients/pow_2_grad/mul_1Mulgradients/pow_2_grad/mulgradients/pow_2_grad/Pow*
T0*
_output_shapes
: 
c
gradients/pow_2_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
q
gradients/pow_2_grad/GreaterGreaterpow_2/xgradients/pow_2_grad/Greater/y*
T0*
_output_shapes
: 
g
$gradients/pow_2_grad/ones_like/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
i
$gradients/pow_2_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
е
gradients/pow_2_grad/ones_likeFill$gradients/pow_2_grad/ones_like/Shape$gradients/pow_2_grad/ones_like/Const*
T0*
_output_shapes
: *

index_type0
Н
gradients/pow_2_grad/SelectSelectgradients/pow_2_grad/Greaterpow_2/xgradients/pow_2_grad/ones_like*
T0*
_output_shapes
: 
]
gradients/pow_2_grad/LogLoggradients/pow_2_grad/Select*
T0*
_output_shapes
: 
d
gradients/pow_2_grad/zeros_likeConst*
_output_shapes
: *
dtype0*
valueB
 *    
б
gradients/pow_2_grad/Select_1Selectgradients/pow_2_grad/Greatergradients/pow_2_grad/Loggradients/pow_2_grad/zeros_like*
T0*
_output_shapes
: 
А
gradients/pow_2_grad/mul_2Mul5gradients/Sum_2/input_grad/tuple/control_dependency_2pow_2*
T0*
_output_shapes
: 
}
gradients/pow_2_grad/mul_3Mulgradients/pow_2_grad/mul_2gradients/pow_2_grad/Select_1*
T0*
_output_shapes
: 
g
%gradients/pow_2_grad/tuple/group_depsNoOp^gradients/pow_2_grad/mul_1^gradients/pow_2_grad/mul_3
═
-gradients/pow_2_grad/tuple/control_dependencyIdentitygradients/pow_2_grad/mul_1&^gradients/pow_2_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_2_grad/mul_1*
_output_shapes
: 
╧
/gradients/pow_2_grad/tuple/control_dependency_1Identitygradients/pow_2_grad/mul_3&^gradients/pow_2_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_2_grad/mul_3*
_output_shapes
: 

gradients/pow_3_grad/mulMul5gradients/Sum_2/input_grad/tuple/control_dependency_3add_27*
T0*
_output_shapes
: 
_
gradients/pow_3_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
d
gradients/pow_3_grad/subSubadd_27gradients/pow_3_grad/sub/y*
T0*
_output_shapes
: 
c
gradients/pow_3_grad/PowPowpow_3/xgradients/pow_3_grad/sub*
T0*
_output_shapes
: 
v
gradients/pow_3_grad/mul_1Mulgradients/pow_3_grad/mulgradients/pow_3_grad/Pow*
T0*
_output_shapes
: 
c
gradients/pow_3_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
q
gradients/pow_3_grad/GreaterGreaterpow_3/xgradients/pow_3_grad/Greater/y*
T0*
_output_shapes
: 
g
$gradients/pow_3_grad/ones_like/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
i
$gradients/pow_3_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
е
gradients/pow_3_grad/ones_likeFill$gradients/pow_3_grad/ones_like/Shape$gradients/pow_3_grad/ones_like/Const*
T0*
_output_shapes
: *

index_type0
Н
gradients/pow_3_grad/SelectSelectgradients/pow_3_grad/Greaterpow_3/xgradients/pow_3_grad/ones_like*
T0*
_output_shapes
: 
]
gradients/pow_3_grad/LogLoggradients/pow_3_grad/Select*
T0*
_output_shapes
: 
d
gradients/pow_3_grad/zeros_likeConst*
_output_shapes
: *
dtype0*
valueB
 *    
б
gradients/pow_3_grad/Select_1Selectgradients/pow_3_grad/Greatergradients/pow_3_grad/Loggradients/pow_3_grad/zeros_like*
T0*
_output_shapes
: 
А
gradients/pow_3_grad/mul_2Mul5gradients/Sum_2/input_grad/tuple/control_dependency_3pow_3*
T0*
_output_shapes
: 
}
gradients/pow_3_grad/mul_3Mulgradients/pow_3_grad/mul_2gradients/pow_3_grad/Select_1*
T0*
_output_shapes
: 
g
%gradients/pow_3_grad/tuple/group_depsNoOp^gradients/pow_3_grad/mul_1^gradients/pow_3_grad/mul_3
═
-gradients/pow_3_grad/tuple/control_dependencyIdentitygradients/pow_3_grad/mul_1&^gradients/pow_3_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_3_grad/mul_1*
_output_shapes
: 
╧
/gradients/pow_3_grad/tuple/control_dependency_1Identitygradients/pow_3_grad/mul_3&^gradients/pow_3_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_3_grad/mul_3*
_output_shapes
: 

gradients/pow_4_grad/mulMul5gradients/Sum_2/input_grad/tuple/control_dependency_4add_34*
T0*
_output_shapes
: 
_
gradients/pow_4_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
d
gradients/pow_4_grad/subSubadd_34gradients/pow_4_grad/sub/y*
T0*
_output_shapes
: 
c
gradients/pow_4_grad/PowPowpow_4/xgradients/pow_4_grad/sub*
T0*
_output_shapes
: 
v
gradients/pow_4_grad/mul_1Mulgradients/pow_4_grad/mulgradients/pow_4_grad/Pow*
T0*
_output_shapes
: 
c
gradients/pow_4_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
q
gradients/pow_4_grad/GreaterGreaterpow_4/xgradients/pow_4_grad/Greater/y*
T0*
_output_shapes
: 
g
$gradients/pow_4_grad/ones_like/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
i
$gradients/pow_4_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
е
gradients/pow_4_grad/ones_likeFill$gradients/pow_4_grad/ones_like/Shape$gradients/pow_4_grad/ones_like/Const*
T0*
_output_shapes
: *

index_type0
Н
gradients/pow_4_grad/SelectSelectgradients/pow_4_grad/Greaterpow_4/xgradients/pow_4_grad/ones_like*
T0*
_output_shapes
: 
]
gradients/pow_4_grad/LogLoggradients/pow_4_grad/Select*
T0*
_output_shapes
: 
d
gradients/pow_4_grad/zeros_likeConst*
_output_shapes
: *
dtype0*
valueB
 *    
б
gradients/pow_4_grad/Select_1Selectgradients/pow_4_grad/Greatergradients/pow_4_grad/Loggradients/pow_4_grad/zeros_like*
T0*
_output_shapes
: 
А
gradients/pow_4_grad/mul_2Mul5gradients/Sum_2/input_grad/tuple/control_dependency_4pow_4*
T0*
_output_shapes
: 
}
gradients/pow_4_grad/mul_3Mulgradients/pow_4_grad/mul_2gradients/pow_4_grad/Select_1*
T0*
_output_shapes
: 
g
%gradients/pow_4_grad/tuple/group_depsNoOp^gradients/pow_4_grad/mul_1^gradients/pow_4_grad/mul_3
═
-gradients/pow_4_grad/tuple/control_dependencyIdentitygradients/pow_4_grad/mul_1&^gradients/pow_4_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_4_grad/mul_1*
_output_shapes
: 
╧
/gradients/pow_4_grad/tuple/control_dependency_1Identitygradients/pow_4_grad/mul_3&^gradients/pow_4_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_4_grad/mul_3*
_output_shapes
: 

gradients/pow_5_grad/mulMul5gradients/Sum_2/input_grad/tuple/control_dependency_5add_41*
T0*
_output_shapes
: 
_
gradients/pow_5_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
d
gradients/pow_5_grad/subSubadd_41gradients/pow_5_grad/sub/y*
T0*
_output_shapes
: 
c
gradients/pow_5_grad/PowPowpow_5/xgradients/pow_5_grad/sub*
T0*
_output_shapes
: 
v
gradients/pow_5_grad/mul_1Mulgradients/pow_5_grad/mulgradients/pow_5_grad/Pow*
T0*
_output_shapes
: 
c
gradients/pow_5_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
q
gradients/pow_5_grad/GreaterGreaterpow_5/xgradients/pow_5_grad/Greater/y*
T0*
_output_shapes
: 
g
$gradients/pow_5_grad/ones_like/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
i
$gradients/pow_5_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
е
gradients/pow_5_grad/ones_likeFill$gradients/pow_5_grad/ones_like/Shape$gradients/pow_5_grad/ones_like/Const*
T0*
_output_shapes
: *

index_type0
Н
gradients/pow_5_grad/SelectSelectgradients/pow_5_grad/Greaterpow_5/xgradients/pow_5_grad/ones_like*
T0*
_output_shapes
: 
]
gradients/pow_5_grad/LogLoggradients/pow_5_grad/Select*
T0*
_output_shapes
: 
d
gradients/pow_5_grad/zeros_likeConst*
_output_shapes
: *
dtype0*
valueB
 *    
б
gradients/pow_5_grad/Select_1Selectgradients/pow_5_grad/Greatergradients/pow_5_grad/Loggradients/pow_5_grad/zeros_like*
T0*
_output_shapes
: 
А
gradients/pow_5_grad/mul_2Mul5gradients/Sum_2/input_grad/tuple/control_dependency_5pow_5*
T0*
_output_shapes
: 
}
gradients/pow_5_grad/mul_3Mulgradients/pow_5_grad/mul_2gradients/pow_5_grad/Select_1*
T0*
_output_shapes
: 
g
%gradients/pow_5_grad/tuple/group_depsNoOp^gradients/pow_5_grad/mul_1^gradients/pow_5_grad/mul_3
═
-gradients/pow_5_grad/tuple/control_dependencyIdentitygradients/pow_5_grad/mul_1&^gradients/pow_5_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_5_grad/mul_1*
_output_shapes
: 
╧
/gradients/pow_5_grad/tuple/control_dependency_1Identitygradients/pow_5_grad/mul_3&^gradients/pow_5_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_5_grad/mul_3*
_output_shapes
: 
_
&gradients/add_43_grad/tuple/group_depsNoOp/^gradients/add_44_grad/tuple/control_dependency
ф
.gradients/add_43_grad/tuple/control_dependencyIdentity.gradients/add_44_grad/tuple/control_dependency'^gradients/add_43_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_14_grad/Mul_1*
_output_shapes
: 
ц
0gradients/add_43_grad/tuple/control_dependency_1Identity.gradients/add_44_grad/tuple/control_dependency'^gradients/add_43_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_14_grad/Mul_1*
_output_shapes
: 
Я
gradients/L2Loss_2_grad/mulMulL2Loss_2/ReadVariableOp0gradients/add_44_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:1
А
б
@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShape!softmax_cross_entropy_with_logits*
T0*
_output_shapes
:*
out_type0
ш
Bgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truediv@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
m
#gradients/Mean_1_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
д
gradients/Mean_1_grad/ReshapeReshape"gradients/Sum_1/input_grad/unstack#gradients/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
И
gradients/Mean_1_grad/ShapeShape-softmax_cross_entropy_with_logits_1/Reshape_2*
T0*
_output_shapes
:*
out_type0
Ю
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:         
К
gradients/Mean_1_grad/Shape_1Shape-softmax_cross_entropy_with_logits_1/Reshape_2*
T0*
_output_shapes
:*
out_type0
`
gradients/Mean_1_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
e
gradients/Mean_1_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Ь
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
g
gradients/Mean_1_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
а
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
a
gradients/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
И
gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
Ж
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
В
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
О
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
T0*#
_output_shapes
:         
]
%gradients/add_6_grad/tuple/group_depsNoOp.^gradients/pow_grad/tuple/control_dependency_1
▐
-gradients/add_6_grad/tuple/control_dependencyIdentity-gradients/pow_grad/tuple/control_dependency_1&^gradients/add_6_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/pow_grad/mul_3*
_output_shapes
: 
р
/gradients/add_6_grad/tuple/control_dependency_1Identity-gradients/pow_grad/tuple/control_dependency_1&^gradients/add_6_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/pow_grad/mul_3*
_output_shapes
: 
`
&gradients/add_13_grad/tuple/group_depsNoOp0^gradients/pow_1_grad/tuple/control_dependency_1
ф
.gradients/add_13_grad/tuple/control_dependencyIdentity/gradients/pow_1_grad/tuple/control_dependency_1'^gradients/add_13_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_1_grad/mul_3*
_output_shapes
: 
ц
0gradients/add_13_grad/tuple/control_dependency_1Identity/gradients/pow_1_grad/tuple/control_dependency_1'^gradients/add_13_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_1_grad/mul_3*
_output_shapes
: 
`
&gradients/add_20_grad/tuple/group_depsNoOp0^gradients/pow_2_grad/tuple/control_dependency_1
ф
.gradients/add_20_grad/tuple/control_dependencyIdentity/gradients/pow_2_grad/tuple/control_dependency_1'^gradients/add_20_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_2_grad/mul_3*
_output_shapes
: 
ц
0gradients/add_20_grad/tuple/control_dependency_1Identity/gradients/pow_2_grad/tuple/control_dependency_1'^gradients/add_20_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_2_grad/mul_3*
_output_shapes
: 
`
&gradients/add_27_grad/tuple/group_depsNoOp0^gradients/pow_3_grad/tuple/control_dependency_1
ф
.gradients/add_27_grad/tuple/control_dependencyIdentity/gradients/pow_3_grad/tuple/control_dependency_1'^gradients/add_27_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_3_grad/mul_3*
_output_shapes
: 
ц
0gradients/add_27_grad/tuple/control_dependency_1Identity/gradients/pow_3_grad/tuple/control_dependency_1'^gradients/add_27_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_3_grad/mul_3*
_output_shapes
: 
`
&gradients/add_34_grad/tuple/group_depsNoOp0^gradients/pow_4_grad/tuple/control_dependency_1
ф
.gradients/add_34_grad/tuple/control_dependencyIdentity/gradients/pow_4_grad/tuple/control_dependency_1'^gradients/add_34_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_4_grad/mul_3*
_output_shapes
: 
ц
0gradients/add_34_grad/tuple/control_dependency_1Identity/gradients/pow_4_grad/tuple/control_dependency_1'^gradients/add_34_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_4_grad/mul_3*
_output_shapes
: 
`
&gradients/add_41_grad/tuple/group_depsNoOp0^gradients/pow_5_grad/tuple/control_dependency_1
ф
.gradients/add_41_grad/tuple/control_dependencyIdentity/gradients/pow_5_grad/tuple/control_dependency_1'^gradients/add_41_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_5_grad/mul_3*
_output_shapes
: 
ц
0gradients/add_41_grad/tuple/control_dependency_1Identity/gradients/pow_5_grad/tuple/control_dependency_1'^gradients/add_41_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_5_grad/mul_3*
_output_shapes
: 
_
&gradients/add_42_grad/tuple/group_depsNoOp/^gradients/add_43_grad/tuple/control_dependency
ф
.gradients/add_42_grad/tuple/control_dependencyIdentity.gradients/add_43_grad/tuple/control_dependency'^gradients/add_42_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_14_grad/Mul_1*
_output_shapes
: 
ц
0gradients/add_42_grad/tuple/control_dependency_1Identity.gradients/add_43_grad/tuple/control_dependency'^gradients/add_42_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_14_grad/Mul_1*
_output_shapes
: 
Ю
gradients/L2Loss_1_grad/mulMulL2Loss_1/ReadVariableOp0gradients/add_43_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:

Б
gradients/zeros_like	ZerosLike#softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:                  
К
?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         
М
;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshape?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:         
╪
4gradients/softmax_cross_entropy_with_logits_grad/mulMul;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims#softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:                  
п
;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax)softmax_cross_entropy_with_logits/Reshape*
T0*0
_output_shapes
:                  
│
4gradients/softmax_cross_entropy_with_logits_grad/NegNeg;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax*
T0*0
_output_shapes
:                  
М
Agradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
         
Р
=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*
T0*

Tdim0*'
_output_shapes
:         
э
6gradients/softmax_cross_entropy_with_logits_grad/mul_1Mul=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_14gradients/softmax_cross_entropy_with_logits_grad/Neg*
T0*0
_output_shapes
:                  
╣
Agradients/softmax_cross_entropy_with_logits_grad/tuple/group_depsNoOp5^gradients/softmax_cross_entropy_with_logits_grad/mul7^gradients/softmax_cross_entropy_with_logits_grad/mul_1
╙
Igradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependencyIdentity4gradients/softmax_cross_entropy_with_logits_grad/mulB^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/softmax_cross_entropy_with_logits_grad/mul*0
_output_shapes
:                  
┘
Kgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency_1Identity6gradients/softmax_cross_entropy_with_logits_grad/mul_1B^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/softmax_cross_entropy_with_logits_grad/mul_1*0
_output_shapes
:                  
е
Bgradients/softmax_cross_entropy_with_logits_1/Reshape_2_grad/ShapeShape#softmax_cross_entropy_with_logits_1*
T0*
_output_shapes
:*
out_type0
ю
Dgradients/softmax_cross_entropy_with_logits_1/Reshape_2_grad/ReshapeReshapegradients/Mean_1_grad/truedivBgradients/softmax_cross_entropy_with_logits_1/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
Ъ
gradients/L2Loss_grad/mulMulL2Loss/ReadVariableOp0gradients/add_42_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:
П
>gradients/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapeflatten_1/Reshape*
T0*
_output_shapes
:*
out_type0
Ц
@gradients/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapeIgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency>gradients/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         

Е
gradients/zeros_like_1	ZerosLike%softmax_cross_entropy_with_logits_1:1*
T0*0
_output_shapes
:                  
М
Agradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         
Т
=gradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims
ExpandDimsDgradients/softmax_cross_entropy_with_logits_1/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:         
▐
6gradients/softmax_cross_entropy_with_logits_1_grad/mulMul=gradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims%softmax_cross_entropy_with_logits_1:1*
T0*0
_output_shapes
:                  
│
=gradients/softmax_cross_entropy_with_logits_1_grad/LogSoftmax
LogSoftmax+softmax_cross_entropy_with_logits_1/Reshape*
T0*0
_output_shapes
:                  
╖
6gradients/softmax_cross_entropy_with_logits_1_grad/NegNeg=gradients/softmax_cross_entropy_with_logits_1_grad/LogSoftmax*
T0*0
_output_shapes
:                  
О
Cgradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
         
Ц
?gradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims_1
ExpandDimsDgradients/softmax_cross_entropy_with_logits_1/Reshape_2_grad/ReshapeCgradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims_1/dim*
T0*

Tdim0*'
_output_shapes
:         
є
8gradients/softmax_cross_entropy_with_logits_1_grad/mul_1Mul?gradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims_16gradients/softmax_cross_entropy_with_logits_1_grad/Neg*
T0*0
_output_shapes
:                  
┐
Cgradients/softmax_cross_entropy_with_logits_1_grad/tuple/group_depsNoOp7^gradients/softmax_cross_entropy_with_logits_1_grad/mul9^gradients/softmax_cross_entropy_with_logits_1_grad/mul_1
█
Kgradients/softmax_cross_entropy_with_logits_1_grad/tuple/control_dependencyIdentity6gradients/softmax_cross_entropy_with_logits_1_grad/mulD^gradients/softmax_cross_entropy_with_logits_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/softmax_cross_entropy_with_logits_1_grad/mul*0
_output_shapes
:                  
с
Mgradients/softmax_cross_entropy_with_logits_1_grad/tuple/control_dependency_1Identity8gradients/softmax_cross_entropy_with_logits_1_grad/mul_1D^gradients/softmax_cross_entropy_with_logits_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/softmax_cross_entropy_with_logits_1_grad/mul_1*0
_output_shapes
:                  
a
gradients/truediv_2_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
c
 gradients/truediv_2_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╞
.gradients/truediv_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_2_grad/Shape gradients/truediv_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Д
 gradients/truediv_2_grad/RealDivRealDiv/gradients/add_6_grad/tuple/control_dependency_1Log_5*
T0*
_output_shapes
: 
│
gradients/truediv_2_grad/SumSum gradients/truediv_2_grad/RealDiv.gradients/truediv_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ш
 gradients/truediv_2_grad/ReshapeReshapegradients/truediv_2_grad/Sumgradients/truediv_2_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
K
gradients/truediv_2_grad/NegNegLog_4*
T0*
_output_shapes
: 
s
"gradients/truediv_2_grad/RealDiv_1RealDivgradients/truediv_2_grad/NegLog_5*
T0*
_output_shapes
: 
y
"gradients/truediv_2_grad/RealDiv_2RealDiv"gradients/truediv_2_grad/RealDiv_1Log_5*
T0*
_output_shapes
: 
Щ
gradients/truediv_2_grad/mulMul/gradients/add_6_grad/tuple/control_dependency_1"gradients/truediv_2_grad/RealDiv_2*
T0*
_output_shapes
: 
│
gradients/truediv_2_grad/Sum_1Sumgradients/truediv_2_grad/mul0gradients/truediv_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ю
"gradients/truediv_2_grad/Reshape_1Reshapegradients/truediv_2_grad/Sum_1 gradients/truediv_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
y
)gradients/truediv_2_grad/tuple/group_depsNoOp!^gradients/truediv_2_grad/Reshape#^gradients/truediv_2_grad/Reshape_1
с
1gradients/truediv_2_grad/tuple/control_dependencyIdentity gradients/truediv_2_grad/Reshape*^gradients/truediv_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_2_grad/Reshape*
_output_shapes
: 
ч
3gradients/truediv_2_grad/tuple/control_dependency_1Identity"gradients/truediv_2_grad/Reshape_1*^gradients/truediv_2_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/truediv_2_grad/Reshape_1*
_output_shapes
: 
a
gradients/truediv_5_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
c
 gradients/truediv_5_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╞
.gradients/truediv_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_5_grad/Shape gradients/truediv_5_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ж
 gradients/truediv_5_grad/RealDivRealDiv0gradients/add_13_grad/tuple/control_dependency_1Log_11*
T0*
_output_shapes
: 
│
gradients/truediv_5_grad/SumSum gradients/truediv_5_grad/RealDiv.gradients/truediv_5_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ш
 gradients/truediv_5_grad/ReshapeReshapegradients/truediv_5_grad/Sumgradients/truediv_5_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
L
gradients/truediv_5_grad/NegNegLog_10*
T0*
_output_shapes
: 
t
"gradients/truediv_5_grad/RealDiv_1RealDivgradients/truediv_5_grad/NegLog_11*
T0*
_output_shapes
: 
z
"gradients/truediv_5_grad/RealDiv_2RealDiv"gradients/truediv_5_grad/RealDiv_1Log_11*
T0*
_output_shapes
: 
Ъ
gradients/truediv_5_grad/mulMul0gradients/add_13_grad/tuple/control_dependency_1"gradients/truediv_5_grad/RealDiv_2*
T0*
_output_shapes
: 
│
gradients/truediv_5_grad/Sum_1Sumgradients/truediv_5_grad/mul0gradients/truediv_5_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ю
"gradients/truediv_5_grad/Reshape_1Reshapegradients/truediv_5_grad/Sum_1 gradients/truediv_5_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
y
)gradients/truediv_5_grad/tuple/group_depsNoOp!^gradients/truediv_5_grad/Reshape#^gradients/truediv_5_grad/Reshape_1
с
1gradients/truediv_5_grad/tuple/control_dependencyIdentity gradients/truediv_5_grad/Reshape*^gradients/truediv_5_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_5_grad/Reshape*
_output_shapes
: 
ч
3gradients/truediv_5_grad/tuple/control_dependency_1Identity"gradients/truediv_5_grad/Reshape_1*^gradients/truediv_5_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/truediv_5_grad/Reshape_1*
_output_shapes
: 
a
gradients/truediv_8_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
c
 gradients/truediv_8_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╞
.gradients/truediv_8_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_8_grad/Shape gradients/truediv_8_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ж
 gradients/truediv_8_grad/RealDivRealDiv0gradients/add_20_grad/tuple/control_dependency_1Log_17*
T0*
_output_shapes
: 
│
gradients/truediv_8_grad/SumSum gradients/truediv_8_grad/RealDiv.gradients/truediv_8_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ш
 gradients/truediv_8_grad/ReshapeReshapegradients/truediv_8_grad/Sumgradients/truediv_8_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
L
gradients/truediv_8_grad/NegNegLog_16*
T0*
_output_shapes
: 
t
"gradients/truediv_8_grad/RealDiv_1RealDivgradients/truediv_8_grad/NegLog_17*
T0*
_output_shapes
: 
z
"gradients/truediv_8_grad/RealDiv_2RealDiv"gradients/truediv_8_grad/RealDiv_1Log_17*
T0*
_output_shapes
: 
Ъ
gradients/truediv_8_grad/mulMul0gradients/add_20_grad/tuple/control_dependency_1"gradients/truediv_8_grad/RealDiv_2*
T0*
_output_shapes
: 
│
gradients/truediv_8_grad/Sum_1Sumgradients/truediv_8_grad/mul0gradients/truediv_8_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ю
"gradients/truediv_8_grad/Reshape_1Reshapegradients/truediv_8_grad/Sum_1 gradients/truediv_8_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
y
)gradients/truediv_8_grad/tuple/group_depsNoOp!^gradients/truediv_8_grad/Reshape#^gradients/truediv_8_grad/Reshape_1
с
1gradients/truediv_8_grad/tuple/control_dependencyIdentity gradients/truediv_8_grad/Reshape*^gradients/truediv_8_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_8_grad/Reshape*
_output_shapes
: 
ч
3gradients/truediv_8_grad/tuple/control_dependency_1Identity"gradients/truediv_8_grad/Reshape_1*^gradients/truediv_8_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/truediv_8_grad/Reshape_1*
_output_shapes
: 
b
gradients/truediv_11_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
d
!gradients/truediv_11_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╔
/gradients/truediv_11_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_11_grad/Shape!gradients/truediv_11_grad/Shape_1*
T0*2
_output_shapes 
:         :         
З
!gradients/truediv_11_grad/RealDivRealDiv0gradients/add_27_grad/tuple/control_dependency_1Log_23*
T0*
_output_shapes
: 
╢
gradients/truediv_11_grad/SumSum!gradients/truediv_11_grad/RealDiv/gradients/truediv_11_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ы
!gradients/truediv_11_grad/ReshapeReshapegradients/truediv_11_grad/Sumgradients/truediv_11_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
M
gradients/truediv_11_grad/NegNegLog_22*
T0*
_output_shapes
: 
v
#gradients/truediv_11_grad/RealDiv_1RealDivgradients/truediv_11_grad/NegLog_23*
T0*
_output_shapes
: 
|
#gradients/truediv_11_grad/RealDiv_2RealDiv#gradients/truediv_11_grad/RealDiv_1Log_23*
T0*
_output_shapes
: 
Ь
gradients/truediv_11_grad/mulMul0gradients/add_27_grad/tuple/control_dependency_1#gradients/truediv_11_grad/RealDiv_2*
T0*
_output_shapes
: 
╢
gradients/truediv_11_grad/Sum_1Sumgradients/truediv_11_grad/mul1gradients/truediv_11_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
б
#gradients/truediv_11_grad/Reshape_1Reshapegradients/truediv_11_grad/Sum_1!gradients/truediv_11_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
*gradients/truediv_11_grad/tuple/group_depsNoOp"^gradients/truediv_11_grad/Reshape$^gradients/truediv_11_grad/Reshape_1
х
2gradients/truediv_11_grad/tuple/control_dependencyIdentity!gradients/truediv_11_grad/Reshape+^gradients/truediv_11_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/truediv_11_grad/Reshape*
_output_shapes
: 
ы
4gradients/truediv_11_grad/tuple/control_dependency_1Identity#gradients/truediv_11_grad/Reshape_1+^gradients/truediv_11_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/truediv_11_grad/Reshape_1*
_output_shapes
: 
b
gradients/truediv_14_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
d
!gradients/truediv_14_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╔
/gradients/truediv_14_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_14_grad/Shape!gradients/truediv_14_grad/Shape_1*
T0*2
_output_shapes 
:         :         
З
!gradients/truediv_14_grad/RealDivRealDiv0gradients/add_34_grad/tuple/control_dependency_1Log_29*
T0*
_output_shapes
: 
╢
gradients/truediv_14_grad/SumSum!gradients/truediv_14_grad/RealDiv/gradients/truediv_14_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ы
!gradients/truediv_14_grad/ReshapeReshapegradients/truediv_14_grad/Sumgradients/truediv_14_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
M
gradients/truediv_14_grad/NegNegLog_28*
T0*
_output_shapes
: 
v
#gradients/truediv_14_grad/RealDiv_1RealDivgradients/truediv_14_grad/NegLog_29*
T0*
_output_shapes
: 
|
#gradients/truediv_14_grad/RealDiv_2RealDiv#gradients/truediv_14_grad/RealDiv_1Log_29*
T0*
_output_shapes
: 
Ь
gradients/truediv_14_grad/mulMul0gradients/add_34_grad/tuple/control_dependency_1#gradients/truediv_14_grad/RealDiv_2*
T0*
_output_shapes
: 
╢
gradients/truediv_14_grad/Sum_1Sumgradients/truediv_14_grad/mul1gradients/truediv_14_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
б
#gradients/truediv_14_grad/Reshape_1Reshapegradients/truediv_14_grad/Sum_1!gradients/truediv_14_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
*gradients/truediv_14_grad/tuple/group_depsNoOp"^gradients/truediv_14_grad/Reshape$^gradients/truediv_14_grad/Reshape_1
х
2gradients/truediv_14_grad/tuple/control_dependencyIdentity!gradients/truediv_14_grad/Reshape+^gradients/truediv_14_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/truediv_14_grad/Reshape*
_output_shapes
: 
ы
4gradients/truediv_14_grad/tuple/control_dependency_1Identity#gradients/truediv_14_grad/Reshape_1+^gradients/truediv_14_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/truediv_14_grad/Reshape_1*
_output_shapes
: 
b
gradients/truediv_17_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
d
!gradients/truediv_17_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╔
/gradients/truediv_17_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_17_grad/Shape!gradients/truediv_17_grad/Shape_1*
T0*2
_output_shapes 
:         :         
З
!gradients/truediv_17_grad/RealDivRealDiv0gradients/add_41_grad/tuple/control_dependency_1Log_35*
T0*
_output_shapes
: 
╢
gradients/truediv_17_grad/SumSum!gradients/truediv_17_grad/RealDiv/gradients/truediv_17_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ы
!gradients/truediv_17_grad/ReshapeReshapegradients/truediv_17_grad/Sumgradients/truediv_17_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
M
gradients/truediv_17_grad/NegNegLog_34*
T0*
_output_shapes
: 
v
#gradients/truediv_17_grad/RealDiv_1RealDivgradients/truediv_17_grad/NegLog_35*
T0*
_output_shapes
: 
|
#gradients/truediv_17_grad/RealDiv_2RealDiv#gradients/truediv_17_grad/RealDiv_1Log_35*
T0*
_output_shapes
: 
Ь
gradients/truediv_17_grad/mulMul0gradients/add_41_grad/tuple/control_dependency_1#gradients/truediv_17_grad/RealDiv_2*
T0*
_output_shapes
: 
╢
gradients/truediv_17_grad/Sum_1Sumgradients/truediv_17_grad/mul1gradients/truediv_17_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
б
#gradients/truediv_17_grad/Reshape_1Reshapegradients/truediv_17_grad/Sum_1!gradients/truediv_17_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
*gradients/truediv_17_grad/tuple/group_depsNoOp"^gradients/truediv_17_grad/Reshape$^gradients/truediv_17_grad/Reshape_1
х
2gradients/truediv_17_grad/tuple/control_dependencyIdentity!gradients/truediv_17_grad/Reshape+^gradients/truediv_17_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/truediv_17_grad/Reshape*
_output_shapes
: 
ы
4gradients/truediv_17_grad/tuple/control_dependency_1Identity#gradients/truediv_17_grad/Reshape_1+^gradients/truediv_17_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/truediv_17_grad/Reshape_1*
_output_shapes
: 
У
@gradients/softmax_cross_entropy_with_logits_1/Reshape_grad/ShapeShapeflatten_1/Reshape_1*
T0*
_output_shapes
:*
out_type0
Ь
Bgradients/softmax_cross_entropy_with_logits_1/Reshape_grad/ReshapeReshapeKgradients/softmax_cross_entropy_with_logits_1_grad/tuple/control_dependency@gradients/softmax_cross_entropy_with_logits_1/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         

Й
Bgradients/softmax_cross_entropy_with_logits_1/Reshape_1_grad/ShapeShapeSoftmax*
T0*
_output_shapes
:*
out_type0
в
Dgradients/softmax_cross_entropy_with_logits_1/Reshape_1_grad/ReshapeReshapeMgradients/softmax_cross_entropy_with_logits_1_grad/tuple/control_dependency_1Bgradients/softmax_cross_entropy_with_logits_1/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         

Й
gradients/Log_4_grad/Reciprocal
Reciprocaladd_52^gradients/truediv_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
Ф
gradients/Log_4_grad/mulMul1gradients/truediv_2_grad/tuple/control_dependencygradients/Log_4_grad/Reciprocal*
T0*
_output_shapes
: 
Л
 gradients/Log_10_grad/Reciprocal
Reciprocaladd_122^gradients/truediv_5_grad/tuple/control_dependency*
T0*
_output_shapes
: 
Ц
gradients/Log_10_grad/mulMul1gradients/truediv_5_grad/tuple/control_dependency gradients/Log_10_grad/Reciprocal*
T0*
_output_shapes
: 
Л
 gradients/Log_16_grad/Reciprocal
Reciprocaladd_192^gradients/truediv_8_grad/tuple/control_dependency*
T0*
_output_shapes
: 
Ц
gradients/Log_16_grad/mulMul1gradients/truediv_8_grad/tuple/control_dependency gradients/Log_16_grad/Reciprocal*
T0*
_output_shapes
: 
М
 gradients/Log_22_grad/Reciprocal
Reciprocaladd_263^gradients/truediv_11_grad/tuple/control_dependency*
T0*
_output_shapes
: 
Ч
gradients/Log_22_grad/mulMul2gradients/truediv_11_grad/tuple/control_dependency gradients/Log_22_grad/Reciprocal*
T0*
_output_shapes
: 
М
 gradients/Log_28_grad/Reciprocal
Reciprocaladd_333^gradients/truediv_14_grad/tuple/control_dependency*
T0*
_output_shapes
: 
Ч
gradients/Log_28_grad/mulMul2gradients/truediv_14_grad/tuple/control_dependency gradients/Log_28_grad/Reciprocal*
T0*
_output_shapes
: 
М
 gradients/Log_34_grad/Reciprocal
Reciprocaladd_403^gradients/truediv_17_grad/tuple/control_dependency*
T0*
_output_shapes
: 
Ч
gradients/Log_34_grad/mulMul2gradients/truediv_17_grad/tuple/control_dependency gradients/Log_34_grad/Reciprocal*
T0*
_output_shapes
: 
y
(gradients/flatten_1/Reshape_1_grad/ShapeShapeconv2d_4/Conv2D_1*
T0*
_output_shapes
:*
out_type0
ы
*gradients/flatten_1/Reshape_1_grad/ReshapeReshapeBgradients/softmax_cross_entropy_with_logits_1/Reshape_grad/Reshape(gradients/flatten_1/Reshape_1_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         

в
gradients/Softmax_grad/mulMulDgradients/softmax_cross_entropy_with_logits_1/Reshape_1_grad/ReshapeSoftmax*
T0*'
_output_shapes
:         

w
,gradients/Softmax_grad/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         
║
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*
T0*

Tidx0*'
_output_shapes
:         *
	keep_dims(
╡
gradients/Softmax_grad/subSubDgradients/softmax_cross_entropy_with_logits_1/Reshape_1_grad/Reshapegradients/Softmax_grad/Sum*
T0*'
_output_shapes
:         

z
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*
T0*'
_output_shapes
:         

H
%gradients/add_5_grad/tuple/group_depsNoOp^gradients/Log_4_grad/mul
╔
-gradients/add_5_grad/tuple/control_dependencyIdentitygradients/Log_4_grad/mul&^gradients/add_5_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Log_4_grad/mul*
_output_shapes
: 
╦
/gradients/add_5_grad/tuple/control_dependency_1Identitygradients/Log_4_grad/mul&^gradients/add_5_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Log_4_grad/mul*
_output_shapes
: 
J
&gradients/add_12_grad/tuple/group_depsNoOp^gradients/Log_10_grad/mul
═
.gradients/add_12_grad/tuple/control_dependencyIdentitygradients/Log_10_grad/mul'^gradients/add_12_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_10_grad/mul*
_output_shapes
: 
╧
0gradients/add_12_grad/tuple/control_dependency_1Identitygradients/Log_10_grad/mul'^gradients/add_12_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_10_grad/mul*
_output_shapes
: 
J
&gradients/add_19_grad/tuple/group_depsNoOp^gradients/Log_16_grad/mul
═
.gradients/add_19_grad/tuple/control_dependencyIdentitygradients/Log_16_grad/mul'^gradients/add_19_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_16_grad/mul*
_output_shapes
: 
╧
0gradients/add_19_grad/tuple/control_dependency_1Identitygradients/Log_16_grad/mul'^gradients/add_19_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_16_grad/mul*
_output_shapes
: 
J
&gradients/add_26_grad/tuple/group_depsNoOp^gradients/Log_22_grad/mul
═
.gradients/add_26_grad/tuple/control_dependencyIdentitygradients/Log_22_grad/mul'^gradients/add_26_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_22_grad/mul*
_output_shapes
: 
╧
0gradients/add_26_grad/tuple/control_dependency_1Identitygradients/Log_22_grad/mul'^gradients/add_26_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_22_grad/mul*
_output_shapes
: 
J
&gradients/add_33_grad/tuple/group_depsNoOp^gradients/Log_28_grad/mul
═
.gradients/add_33_grad/tuple/control_dependencyIdentitygradients/Log_28_grad/mul'^gradients/add_33_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_28_grad/mul*
_output_shapes
: 
╧
0gradients/add_33_grad/tuple/control_dependency_1Identitygradients/Log_28_grad/mul'^gradients/add_33_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_28_grad/mul*
_output_shapes
: 
J
&gradients/add_40_grad/tuple/group_depsNoOp^gradients/Log_34_grad/mul
═
.gradients/add_40_grad/tuple/control_dependencyIdentitygradients/Log_34_grad/mul'^gradients/add_40_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_34_grad/mul*
_output_shapes
: 
╧
0gradients/add_40_grad/tuple/control_dependency_1Identitygradients/Log_34_grad/mul'^gradients/add_40_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_34_grad/mul*
_output_shapes
: 
е
'gradients/conv2d_4/Conv2D_1_grad/ShapeNShapeNreshape_2/reshape_2_requantizeconv2d_4/mul_2*
N*
T0* 
_output_shapes
::*
out_type0
є
4gradients/conv2d_4/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/conv2d_4/Conv2D_1_grad/ShapeNconv2d_4/mul_2*gradients/flatten_1/Reshape_1_grad/Reshape*
T0*0
_output_shapes
:         А*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
■
5gradients/conv2d_4/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterreshape_2/reshape_2_requantize)gradients/conv2d_4/Conv2D_1_grad/ShapeN:1*gradients/flatten_1/Reshape_1_grad/Reshape*
T0*'
_output_shapes
:А
*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
и
1gradients/conv2d_4/Conv2D_1_grad/tuple/group_depsNoOp6^gradients/conv2d_4/Conv2D_1_grad/Conv2DBackpropFilter5^gradients/conv2d_4/Conv2D_1_grad/Conv2DBackpropInput
│
9gradients/conv2d_4/Conv2D_1_grad/tuple/control_dependencyIdentity4gradients/conv2d_4/Conv2D_1_grad/Conv2DBackpropInput2^gradients/conv2d_4/Conv2D_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/conv2d_4/Conv2D_1_grad/Conv2DBackpropInput*0
_output_shapes
:         А
о
;gradients/conv2d_4/Conv2D_1_grad/tuple/control_dependency_1Identity5gradients/conv2d_4/Conv2D_1_grad/Conv2DBackpropFilter2^gradients/conv2d_4/Conv2D_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/conv2d_4/Conv2D_1_grad/Conv2DBackpropFilter*'
_output_shapes
:А

Ж
gradients/AddNAddN@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshapegradients/Softmax_grad/mul_1*
N*
T0*S
_classI
GEloc:@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*'
_output_shapes
:         

u
&gradients/flatten_1/Reshape_grad/ShapeShapeconv2d_4/Conv2D*
T0*
_output_shapes
:*
out_type0
│
(gradients/flatten_1/Reshape_grad/ReshapeReshapegradients/AddN&gradients/flatten_1/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         

I
gradients/Abs_2_grad/SignSignadd_4*
T0*
_output_shapes
: 
К
gradients/Abs_2_grad/mulMul-gradients/add_5_grad/tuple/control_dependencygradients/Abs_2_grad/Sign*
T0*
_output_shapes
: 
J
gradients/Abs_5_grad/SignSignadd_11*
T0*
_output_shapes
: 
Л
gradients/Abs_5_grad/mulMul.gradients/add_12_grad/tuple/control_dependencygradients/Abs_5_grad/Sign*
T0*
_output_shapes
: 
J
gradients/Abs_8_grad/SignSignadd_18*
T0*
_output_shapes
: 
Л
gradients/Abs_8_grad/mulMul.gradients/add_19_grad/tuple/control_dependencygradients/Abs_8_grad/Sign*
T0*
_output_shapes
: 
K
gradients/Abs_11_grad/SignSignadd_25*
T0*
_output_shapes
: 
Н
gradients/Abs_11_grad/mulMul.gradients/add_26_grad/tuple/control_dependencygradients/Abs_11_grad/Sign*
T0*
_output_shapes
: 
K
gradients/Abs_14_grad/SignSignadd_32*
T0*
_output_shapes
: 
Н
gradients/Abs_14_grad/mulMul.gradients/add_33_grad/tuple/control_dependencygradients/Abs_14_grad/Sign*
T0*
_output_shapes
: 
K
gradients/Abs_17_grad/SignSignadd_39*
T0*
_output_shapes
: 
Н
gradients/Abs_17_grad/mulMul.gradients/add_40_grad/tuple/control_dependencygradients/Abs_17_grad/Sign*
T0*
_output_shapes
: 
ж
%gradients/conv2d_4/Conv2D_grad/ShapeNShapeNreshape_2/Reshapeconv2d_4/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::*
out_type0
¤
2gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_4/Conv2D_grad/ShapeNconv2d_4/Conv2D/ReadVariableOp(gradients/flatten_1/Reshape_grad/Reshape*
T0*0
_output_shapes
:         А*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
ы
3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterreshape_2/Reshape'gradients/conv2d_4/Conv2D_grad/ShapeN:1(gradients/flatten_1/Reshape_grad/Reshape*
T0*'
_output_shapes
:А
*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
в
/gradients/conv2d_4/Conv2D_grad/tuple/group_depsNoOp4^gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter3^gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput
л
7gradients/conv2d_4/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_4/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         А
ж
9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_4/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:А

H
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/Abs_2_grad/mul
╔
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/Abs_2_grad/mul&^gradients/add_4_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Abs_2_grad/mul*
_output_shapes
: 
╦
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/Abs_2_grad/mul&^gradients/add_4_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Abs_2_grad/mul*
_output_shapes
: 
I
&gradients/add_11_grad/tuple/group_depsNoOp^gradients/Abs_5_grad/mul
╦
.gradients/add_11_grad/tuple/control_dependencyIdentitygradients/Abs_5_grad/mul'^gradients/add_11_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Abs_5_grad/mul*
_output_shapes
: 
═
0gradients/add_11_grad/tuple/control_dependency_1Identitygradients/Abs_5_grad/mul'^gradients/add_11_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Abs_5_grad/mul*
_output_shapes
: 
I
&gradients/add_18_grad/tuple/group_depsNoOp^gradients/Abs_8_grad/mul
╦
.gradients/add_18_grad/tuple/control_dependencyIdentitygradients/Abs_8_grad/mul'^gradients/add_18_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Abs_8_grad/mul*
_output_shapes
: 
═
0gradients/add_18_grad/tuple/control_dependency_1Identitygradients/Abs_8_grad/mul'^gradients/add_18_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Abs_8_grad/mul*
_output_shapes
: 
J
&gradients/add_25_grad/tuple/group_depsNoOp^gradients/Abs_11_grad/mul
═
.gradients/add_25_grad/tuple/control_dependencyIdentitygradients/Abs_11_grad/mul'^gradients/add_25_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Abs_11_grad/mul*
_output_shapes
: 
╧
0gradients/add_25_grad/tuple/control_dependency_1Identitygradients/Abs_11_grad/mul'^gradients/add_25_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Abs_11_grad/mul*
_output_shapes
: 
J
&gradients/add_32_grad/tuple/group_depsNoOp^gradients/Abs_14_grad/mul
═
.gradients/add_32_grad/tuple/control_dependencyIdentitygradients/Abs_14_grad/mul'^gradients/add_32_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Abs_14_grad/mul*
_output_shapes
: 
╧
0gradients/add_32_grad/tuple/control_dependency_1Identitygradients/Abs_14_grad/mul'^gradients/add_32_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Abs_14_grad/mul*
_output_shapes
: 
J
&gradients/add_39_grad/tuple/group_depsNoOp^gradients/Abs_17_grad/mul
═
.gradients/add_39_grad/tuple/control_dependencyIdentitygradients/Abs_17_grad/mul'^gradients/add_39_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Abs_17_grad/mul*
_output_shapes
: 
╧
0gradients/add_39_grad/tuple/control_dependency_1Identitygradients/Abs_17_grad/mul'^gradients/add_39_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Abs_17_grad/mul*
_output_shapes
: 
}
(gradients/reshape_2/Reshape_1_grad/ShapeShapeactivation_4/Identity*
T0*
_output_shapes
:*
out_type0
у
*gradients/reshape_2/Reshape_1_grad/ReshapeReshape9gradients/conv2d_4/Conv2D_1_grad/tuple/control_dependency(gradients/reshape_2/Reshape_1_grad/Shape*
T0*
Tshape0*0
_output_shapes
:         А
w
&gradients/reshape_2/Reshape_grad/ShapeShapeactivation_4/Relu*
T0*
_output_shapes
:*
out_type0
▌
(gradients/reshape_2/Reshape_grad/ReshapeReshape7gradients/conv2d_4/Conv2D_grad/tuple/control_dependency&gradients/reshape_2/Reshape_grad/Shape*
T0*
Tshape0*0
_output_shapes
:         А
m
gradients/sub_grad/NegNeg-gradients/add_4_grad/tuple/control_dependency*
T0*
_output_shapes
: 
t
#gradients/sub_grad/tuple/group_depsNoOp.^gradients/add_4_grad/tuple/control_dependency^gradients/sub_grad/Neg
┌
+gradients/sub_grad/tuple/control_dependencyIdentity-gradients/add_4_grad/tuple/control_dependency$^gradients/sub_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Abs_2_grad/mul*
_output_shapes
: 
├
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Neg$^gradients/sub_grad/tuple/group_deps*
T0*)
_class
loc:@gradients/sub_grad/Neg*
_output_shapes
: 
p
gradients/sub_1_grad/NegNeg.gradients/add_11_grad/tuple/control_dependency*
T0*
_output_shapes
: 
y
%gradients/sub_1_grad/tuple/group_depsNoOp/^gradients/add_11_grad/tuple/control_dependency^gradients/sub_1_grad/Neg
▀
-gradients/sub_1_grad/tuple/control_dependencyIdentity.gradients/add_11_grad/tuple/control_dependency&^gradients/sub_1_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Abs_5_grad/mul*
_output_shapes
: 
╦
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Neg&^gradients/sub_1_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/sub_1_grad/Neg*
_output_shapes
: 
p
gradients/sub_2_grad/NegNeg.gradients/add_18_grad/tuple/control_dependency*
T0*
_output_shapes
: 
y
%gradients/sub_2_grad/tuple/group_depsNoOp/^gradients/add_18_grad/tuple/control_dependency^gradients/sub_2_grad/Neg
▀
-gradients/sub_2_grad/tuple/control_dependencyIdentity.gradients/add_18_grad/tuple/control_dependency&^gradients/sub_2_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Abs_8_grad/mul*
_output_shapes
: 
╦
/gradients/sub_2_grad/tuple/control_dependency_1Identitygradients/sub_2_grad/Neg&^gradients/sub_2_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/sub_2_grad/Neg*
_output_shapes
: 
p
gradients/sub_3_grad/NegNeg.gradients/add_25_grad/tuple/control_dependency*
T0*
_output_shapes
: 
y
%gradients/sub_3_grad/tuple/group_depsNoOp/^gradients/add_25_grad/tuple/control_dependency^gradients/sub_3_grad/Neg
р
-gradients/sub_3_grad/tuple/control_dependencyIdentity.gradients/add_25_grad/tuple/control_dependency&^gradients/sub_3_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Abs_11_grad/mul*
_output_shapes
: 
╦
/gradients/sub_3_grad/tuple/control_dependency_1Identitygradients/sub_3_grad/Neg&^gradients/sub_3_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/sub_3_grad/Neg*
_output_shapes
: 
p
gradients/sub_4_grad/NegNeg.gradients/add_32_grad/tuple/control_dependency*
T0*
_output_shapes
: 
y
%gradients/sub_4_grad/tuple/group_depsNoOp/^gradients/add_32_grad/tuple/control_dependency^gradients/sub_4_grad/Neg
р
-gradients/sub_4_grad/tuple/control_dependencyIdentity.gradients/add_32_grad/tuple/control_dependency&^gradients/sub_4_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Abs_14_grad/mul*
_output_shapes
: 
╦
/gradients/sub_4_grad/tuple/control_dependency_1Identitygradients/sub_4_grad/Neg&^gradients/sub_4_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/sub_4_grad/Neg*
_output_shapes
: 
p
gradients/sub_5_grad/NegNeg.gradients/add_39_grad/tuple/control_dependency*
T0*
_output_shapes
: 
y
%gradients/sub_5_grad/tuple/group_depsNoOp/^gradients/add_39_grad/tuple/control_dependency^gradients/sub_5_grad/Neg
р
-gradients/sub_5_grad/tuple/control_dependencyIdentity.gradients/add_39_grad/tuple/control_dependency&^gradients/sub_5_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Abs_17_grad/mul*
_output_shapes
: 
╦
/gradients/sub_5_grad/tuple/control_dependency_1Identitygradients/sub_5_grad/Neg&^gradients/sub_5_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/sub_5_grad/Neg*
_output_shapes
: 
н
)gradients/activation_4/Relu_grad/ReluGradReluGrad(gradients/reshape_2/Reshape_grad/Reshapeactivation_4/Relu*
T0*0
_output_shapes
:         А
d
gradients/Max_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
[
gradients/Max_1_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
j
gradients/Max_1_grad/addAddV2range_3gradients/Max_1_grad/Size*
T0*
_output_shapes
:
~
gradients/Max_1_grad/modFloorModgradients/Max_1_grad/addgradients/Max_1_grad/Size*
T0*
_output_shapes
:
f
gradients/Max_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Max_1_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Max_1_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Max_1_grad/rangeRange gradients/Max_1_grad/range/startgradients/Max_1_grad/Size gradients/Max_1_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Max_1_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Max_1_grad/FillFillgradients/Max_1_grad/Shape_1gradients/Max_1_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Max_1_grad/DynamicStitchDynamicStitchgradients/Max_1_grad/rangegradients/Max_1_grad/modgradients/Max_1_grad/Shapegradients/Max_1_grad/Fill*
N*
T0*
_output_shapes
:
Е
gradients/Max_1_grad/ReshapeReshapeMax_1"gradients/Max_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
н
gradients/Max_1_grad/Reshape_1Reshape+gradients/sub_grad/tuple/control_dependency"gradients/Max_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
У
gradients/Max_1_grad/EqualEqualgradients/Max_1_grad/ReshapeMax_1/input*
T0*
_output_shapes
:*
incompatible_shape_error(
Б
gradients/Max_1_grad/CastCastgradients/Max_1_grad/Equal*

DstT0*

SrcT0
*
Truncate( *
_output_shapes
:
Б
gradients/Max_1_grad/SumSumgradients/Max_1_grad/Castrange_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ъ
gradients/Max_1_grad/Reshape_2Reshapegradients/Max_1_grad/Sum"gradients/Max_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
З
gradients/Max_1_grad/truedivRealDivgradients/Max_1_grad/Castgradients/Max_1_grad/Reshape_2*
T0*
_output_shapes
:
В
gradients/Max_1_grad/mulMulgradients/Max_1_grad/truedivgradients/Max_1_grad/Reshape_1*
T0*
_output_shapes
:
d
gradients/Min_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
[
gradients/Min_1_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
j
gradients/Min_1_grad/addAddV2range_2gradients/Min_1_grad/Size*
T0*
_output_shapes
:
~
gradients/Min_1_grad/modFloorModgradients/Min_1_grad/addgradients/Min_1_grad/Size*
T0*
_output_shapes
:
f
gradients/Min_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Min_1_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Min_1_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Min_1_grad/rangeRange gradients/Min_1_grad/range/startgradients/Min_1_grad/Size gradients/Min_1_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Min_1_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Min_1_grad/FillFillgradients/Min_1_grad/Shape_1gradients/Min_1_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Min_1_grad/DynamicStitchDynamicStitchgradients/Min_1_grad/rangegradients/Min_1_grad/modgradients/Min_1_grad/Shapegradients/Min_1_grad/Fill*
N*
T0*
_output_shapes
:
Е
gradients/Min_1_grad/ReshapeReshapeMin_1"gradients/Min_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
п
gradients/Min_1_grad/Reshape_1Reshape-gradients/sub_grad/tuple/control_dependency_1"gradients/Min_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
У
gradients/Min_1_grad/EqualEqualgradients/Min_1_grad/ReshapeMin_1/input*
T0*
_output_shapes
:*
incompatible_shape_error(
Б
gradients/Min_1_grad/CastCastgradients/Min_1_grad/Equal*

DstT0*

SrcT0
*
Truncate( *
_output_shapes
:
Б
gradients/Min_1_grad/SumSumgradients/Min_1_grad/Castrange_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ъ
gradients/Min_1_grad/Reshape_2Reshapegradients/Min_1_grad/Sum"gradients/Min_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
З
gradients/Min_1_grad/truedivRealDivgradients/Min_1_grad/Castgradients/Min_1_grad/Reshape_2*
T0*
_output_shapes
:
В
gradients/Min_1_grad/mulMulgradients/Min_1_grad/truedivgradients/Min_1_grad/Reshape_1*
T0*
_output_shapes
:
d
gradients/Max_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
[
gradients/Max_3_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
j
gradients/Max_3_grad/addAddV2range_5gradients/Max_3_grad/Size*
T0*
_output_shapes
:
~
gradients/Max_3_grad/modFloorModgradients/Max_3_grad/addgradients/Max_3_grad/Size*
T0*
_output_shapes
:
f
gradients/Max_3_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Max_3_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Max_3_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Max_3_grad/rangeRange gradients/Max_3_grad/range/startgradients/Max_3_grad/Size gradients/Max_3_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Max_3_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Max_3_grad/FillFillgradients/Max_3_grad/Shape_1gradients/Max_3_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Max_3_grad/DynamicStitchDynamicStitchgradients/Max_3_grad/rangegradients/Max_3_grad/modgradients/Max_3_grad/Shapegradients/Max_3_grad/Fill*
N*
T0*
_output_shapes
:
Е
gradients/Max_3_grad/ReshapeReshapeMax_3"gradients/Max_3_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
п
gradients/Max_3_grad/Reshape_1Reshape-gradients/sub_1_grad/tuple/control_dependency"gradients/Max_3_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
У
gradients/Max_3_grad/EqualEqualgradients/Max_3_grad/ReshapeMax_3/input*
T0*
_output_shapes
:*
incompatible_shape_error(
Б
gradients/Max_3_grad/CastCastgradients/Max_3_grad/Equal*

DstT0*

SrcT0
*
Truncate( *
_output_shapes
:
Б
gradients/Max_3_grad/SumSumgradients/Max_3_grad/Castrange_5*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ъ
gradients/Max_3_grad/Reshape_2Reshapegradients/Max_3_grad/Sum"gradients/Max_3_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
З
gradients/Max_3_grad/truedivRealDivgradients/Max_3_grad/Castgradients/Max_3_grad/Reshape_2*
T0*
_output_shapes
:
В
gradients/Max_3_grad/mulMulgradients/Max_3_grad/truedivgradients/Max_3_grad/Reshape_1*
T0*
_output_shapes
:
d
gradients/Min_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
[
gradients/Min_3_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
j
gradients/Min_3_grad/addAddV2range_4gradients/Min_3_grad/Size*
T0*
_output_shapes
:
~
gradients/Min_3_grad/modFloorModgradients/Min_3_grad/addgradients/Min_3_grad/Size*
T0*
_output_shapes
:
f
gradients/Min_3_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Min_3_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Min_3_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Min_3_grad/rangeRange gradients/Min_3_grad/range/startgradients/Min_3_grad/Size gradients/Min_3_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Min_3_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Min_3_grad/FillFillgradients/Min_3_grad/Shape_1gradients/Min_3_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Min_3_grad/DynamicStitchDynamicStitchgradients/Min_3_grad/rangegradients/Min_3_grad/modgradients/Min_3_grad/Shapegradients/Min_3_grad/Fill*
N*
T0*
_output_shapes
:
Е
gradients/Min_3_grad/ReshapeReshapeMin_3"gradients/Min_3_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
▒
gradients/Min_3_grad/Reshape_1Reshape/gradients/sub_1_grad/tuple/control_dependency_1"gradients/Min_3_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
У
gradients/Min_3_grad/EqualEqualgradients/Min_3_grad/ReshapeMin_3/input*
T0*
_output_shapes
:*
incompatible_shape_error(
Б
gradients/Min_3_grad/CastCastgradients/Min_3_grad/Equal*

DstT0*

SrcT0
*
Truncate( *
_output_shapes
:
Б
gradients/Min_3_grad/SumSumgradients/Min_3_grad/Castrange_4*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ъ
gradients/Min_3_grad/Reshape_2Reshapegradients/Min_3_grad/Sum"gradients/Min_3_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
З
gradients/Min_3_grad/truedivRealDivgradients/Min_3_grad/Castgradients/Min_3_grad/Reshape_2*
T0*
_output_shapes
:
В
gradients/Min_3_grad/mulMulgradients/Min_3_grad/truedivgradients/Min_3_grad/Reshape_1*
T0*
_output_shapes
:
d
gradients/Max_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
[
gradients/Max_5_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
j
gradients/Max_5_grad/addAddV2range_7gradients/Max_5_grad/Size*
T0*
_output_shapes
:
~
gradients/Max_5_grad/modFloorModgradients/Max_5_grad/addgradients/Max_5_grad/Size*
T0*
_output_shapes
:
f
gradients/Max_5_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Max_5_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Max_5_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Max_5_grad/rangeRange gradients/Max_5_grad/range/startgradients/Max_5_grad/Size gradients/Max_5_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Max_5_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Max_5_grad/FillFillgradients/Max_5_grad/Shape_1gradients/Max_5_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Max_5_grad/DynamicStitchDynamicStitchgradients/Max_5_grad/rangegradients/Max_5_grad/modgradients/Max_5_grad/Shapegradients/Max_5_grad/Fill*
N*
T0*
_output_shapes
:
Е
gradients/Max_5_grad/ReshapeReshapeMax_5"gradients/Max_5_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
п
gradients/Max_5_grad/Reshape_1Reshape-gradients/sub_2_grad/tuple/control_dependency"gradients/Max_5_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
У
gradients/Max_5_grad/EqualEqualgradients/Max_5_grad/ReshapeMax_5/input*
T0*
_output_shapes
:*
incompatible_shape_error(
Б
gradients/Max_5_grad/CastCastgradients/Max_5_grad/Equal*

DstT0*

SrcT0
*
Truncate( *
_output_shapes
:
Б
gradients/Max_5_grad/SumSumgradients/Max_5_grad/Castrange_7*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ъ
gradients/Max_5_grad/Reshape_2Reshapegradients/Max_5_grad/Sum"gradients/Max_5_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
З
gradients/Max_5_grad/truedivRealDivgradients/Max_5_grad/Castgradients/Max_5_grad/Reshape_2*
T0*
_output_shapes
:
В
gradients/Max_5_grad/mulMulgradients/Max_5_grad/truedivgradients/Max_5_grad/Reshape_1*
T0*
_output_shapes
:
d
gradients/Min_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
[
gradients/Min_5_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
j
gradients/Min_5_grad/addAddV2range_6gradients/Min_5_grad/Size*
T0*
_output_shapes
:
~
gradients/Min_5_grad/modFloorModgradients/Min_5_grad/addgradients/Min_5_grad/Size*
T0*
_output_shapes
:
f
gradients/Min_5_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Min_5_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Min_5_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Min_5_grad/rangeRange gradients/Min_5_grad/range/startgradients/Min_5_grad/Size gradients/Min_5_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Min_5_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Min_5_grad/FillFillgradients/Min_5_grad/Shape_1gradients/Min_5_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Min_5_grad/DynamicStitchDynamicStitchgradients/Min_5_grad/rangegradients/Min_5_grad/modgradients/Min_5_grad/Shapegradients/Min_5_grad/Fill*
N*
T0*
_output_shapes
:
Е
gradients/Min_5_grad/ReshapeReshapeMin_5"gradients/Min_5_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
▒
gradients/Min_5_grad/Reshape_1Reshape/gradients/sub_2_grad/tuple/control_dependency_1"gradients/Min_5_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
У
gradients/Min_5_grad/EqualEqualgradients/Min_5_grad/ReshapeMin_5/input*
T0*
_output_shapes
:*
incompatible_shape_error(
Б
gradients/Min_5_grad/CastCastgradients/Min_5_grad/Equal*

DstT0*

SrcT0
*
Truncate( *
_output_shapes
:
Б
gradients/Min_5_grad/SumSumgradients/Min_5_grad/Castrange_6*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ъ
gradients/Min_5_grad/Reshape_2Reshapegradients/Min_5_grad/Sum"gradients/Min_5_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
З
gradients/Min_5_grad/truedivRealDivgradients/Min_5_grad/Castgradients/Min_5_grad/Reshape_2*
T0*
_output_shapes
:
В
gradients/Min_5_grad/mulMulgradients/Min_5_grad/truedivgradients/Min_5_grad/Reshape_1*
T0*
_output_shapes
:
d
gradients/Max_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
[
gradients/Max_7_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
j
gradients/Max_7_grad/addAddV2range_9gradients/Max_7_grad/Size*
T0*
_output_shapes
:
~
gradients/Max_7_grad/modFloorModgradients/Max_7_grad/addgradients/Max_7_grad/Size*
T0*
_output_shapes
:
f
gradients/Max_7_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Max_7_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Max_7_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Max_7_grad/rangeRange gradients/Max_7_grad/range/startgradients/Max_7_grad/Size gradients/Max_7_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Max_7_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Max_7_grad/FillFillgradients/Max_7_grad/Shape_1gradients/Max_7_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Max_7_grad/DynamicStitchDynamicStitchgradients/Max_7_grad/rangegradients/Max_7_grad/modgradients/Max_7_grad/Shapegradients/Max_7_grad/Fill*
N*
T0*
_output_shapes
:
Е
gradients/Max_7_grad/ReshapeReshapeMax_7"gradients/Max_7_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
п
gradients/Max_7_grad/Reshape_1Reshape-gradients/sub_3_grad/tuple/control_dependency"gradients/Max_7_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
У
gradients/Max_7_grad/EqualEqualgradients/Max_7_grad/ReshapeMax_7/input*
T0*
_output_shapes
:*
incompatible_shape_error(
Б
gradients/Max_7_grad/CastCastgradients/Max_7_grad/Equal*

DstT0*

SrcT0
*
Truncate( *
_output_shapes
:
Б
gradients/Max_7_grad/SumSumgradients/Max_7_grad/Castrange_9*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ъ
gradients/Max_7_grad/Reshape_2Reshapegradients/Max_7_grad/Sum"gradients/Max_7_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
З
gradients/Max_7_grad/truedivRealDivgradients/Max_7_grad/Castgradients/Max_7_grad/Reshape_2*
T0*
_output_shapes
:
В
gradients/Max_7_grad/mulMulgradients/Max_7_grad/truedivgradients/Max_7_grad/Reshape_1*
T0*
_output_shapes
:
d
gradients/Min_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
[
gradients/Min_7_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
j
gradients/Min_7_grad/addAddV2range_8gradients/Min_7_grad/Size*
T0*
_output_shapes
:
~
gradients/Min_7_grad/modFloorModgradients/Min_7_grad/addgradients/Min_7_grad/Size*
T0*
_output_shapes
:
f
gradients/Min_7_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Min_7_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Min_7_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Min_7_grad/rangeRange gradients/Min_7_grad/range/startgradients/Min_7_grad/Size gradients/Min_7_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Min_7_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Min_7_grad/FillFillgradients/Min_7_grad/Shape_1gradients/Min_7_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Min_7_grad/DynamicStitchDynamicStitchgradients/Min_7_grad/rangegradients/Min_7_grad/modgradients/Min_7_grad/Shapegradients/Min_7_grad/Fill*
N*
T0*
_output_shapes
:
Е
gradients/Min_7_grad/ReshapeReshapeMin_7"gradients/Min_7_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
▒
gradients/Min_7_grad/Reshape_1Reshape/gradients/sub_3_grad/tuple/control_dependency_1"gradients/Min_7_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
У
gradients/Min_7_grad/EqualEqualgradients/Min_7_grad/ReshapeMin_7/input*
T0*
_output_shapes
:*
incompatible_shape_error(
Б
gradients/Min_7_grad/CastCastgradients/Min_7_grad/Equal*

DstT0*

SrcT0
*
Truncate( *
_output_shapes
:
Б
gradients/Min_7_grad/SumSumgradients/Min_7_grad/Castrange_8*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ъ
gradients/Min_7_grad/Reshape_2Reshapegradients/Min_7_grad/Sum"gradients/Min_7_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
З
gradients/Min_7_grad/truedivRealDivgradients/Min_7_grad/Castgradients/Min_7_grad/Reshape_2*
T0*
_output_shapes
:
В
gradients/Min_7_grad/mulMulgradients/Min_7_grad/truedivgradients/Min_7_grad/Reshape_1*
T0*
_output_shapes
:
d
gradients/Max_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
[
gradients/Max_9_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
k
gradients/Max_9_grad/addAddV2range_11gradients/Max_9_grad/Size*
T0*
_output_shapes
:
~
gradients/Max_9_grad/modFloorModgradients/Max_9_grad/addgradients/Max_9_grad/Size*
T0*
_output_shapes
:
f
gradients/Max_9_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Max_9_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Max_9_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Max_9_grad/rangeRange gradients/Max_9_grad/range/startgradients/Max_9_grad/Size gradients/Max_9_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Max_9_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Max_9_grad/FillFillgradients/Max_9_grad/Shape_1gradients/Max_9_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Max_9_grad/DynamicStitchDynamicStitchgradients/Max_9_grad/rangegradients/Max_9_grad/modgradients/Max_9_grad/Shapegradients/Max_9_grad/Fill*
N*
T0*
_output_shapes
:
Е
gradients/Max_9_grad/ReshapeReshapeMax_9"gradients/Max_9_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
п
gradients/Max_9_grad/Reshape_1Reshape-gradients/sub_4_grad/tuple/control_dependency"gradients/Max_9_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
У
gradients/Max_9_grad/EqualEqualgradients/Max_9_grad/ReshapeMax_9/input*
T0*
_output_shapes
:*
incompatible_shape_error(
Б
gradients/Max_9_grad/CastCastgradients/Max_9_grad/Equal*

DstT0*

SrcT0
*
Truncate( *
_output_shapes
:
В
gradients/Max_9_grad/SumSumgradients/Max_9_grad/Castrange_11*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ъ
gradients/Max_9_grad/Reshape_2Reshapegradients/Max_9_grad/Sum"gradients/Max_9_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
З
gradients/Max_9_grad/truedivRealDivgradients/Max_9_grad/Castgradients/Max_9_grad/Reshape_2*
T0*
_output_shapes
:
В
gradients/Max_9_grad/mulMulgradients/Max_9_grad/truedivgradients/Max_9_grad/Reshape_1*
T0*
_output_shapes
:
d
gradients/Min_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
[
gradients/Min_9_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
k
gradients/Min_9_grad/addAddV2range_10gradients/Min_9_grad/Size*
T0*
_output_shapes
:
~
gradients/Min_9_grad/modFloorModgradients/Min_9_grad/addgradients/Min_9_grad/Size*
T0*
_output_shapes
:
f
gradients/Min_9_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Min_9_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Min_9_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Min_9_grad/rangeRange gradients/Min_9_grad/range/startgradients/Min_9_grad/Size gradients/Min_9_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Min_9_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Min_9_grad/FillFillgradients/Min_9_grad/Shape_1gradients/Min_9_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Min_9_grad/DynamicStitchDynamicStitchgradients/Min_9_grad/rangegradients/Min_9_grad/modgradients/Min_9_grad/Shapegradients/Min_9_grad/Fill*
N*
T0*
_output_shapes
:
Е
gradients/Min_9_grad/ReshapeReshapeMin_9"gradients/Min_9_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
▒
gradients/Min_9_grad/Reshape_1Reshape/gradients/sub_4_grad/tuple/control_dependency_1"gradients/Min_9_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
У
gradients/Min_9_grad/EqualEqualgradients/Min_9_grad/ReshapeMin_9/input*
T0*
_output_shapes
:*
incompatible_shape_error(
Б
gradients/Min_9_grad/CastCastgradients/Min_9_grad/Equal*

DstT0*

SrcT0
*
Truncate( *
_output_shapes
:
В
gradients/Min_9_grad/SumSumgradients/Min_9_grad/Castrange_10*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ъ
gradients/Min_9_grad/Reshape_2Reshapegradients/Min_9_grad/Sum"gradients/Min_9_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
З
gradients/Min_9_grad/truedivRealDivgradients/Min_9_grad/Castgradients/Min_9_grad/Reshape_2*
T0*
_output_shapes
:
В
gradients/Min_9_grad/mulMulgradients/Min_9_grad/truedivgradients/Min_9_grad/Reshape_1*
T0*
_output_shapes
:
e
gradients/Max_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
\
gradients/Max_11_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
m
gradients/Max_11_grad/addAddV2range_13gradients/Max_11_grad/Size*
T0*
_output_shapes
:
Б
gradients/Max_11_grad/modFloorModgradients/Max_11_grad/addgradients/Max_11_grad/Size*
T0*
_output_shapes
:
g
gradients/Max_11_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
c
!gradients/Max_11_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
c
!gradients/Max_11_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
о
gradients/Max_11_grad/rangeRange!gradients/Max_11_grad/range/startgradients/Max_11_grad/Size!gradients/Max_11_grad/range/delta*

Tidx0*
_output_shapes
:
b
 gradients/Max_11_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ъ
gradients/Max_11_grad/FillFillgradients/Max_11_grad/Shape_1 gradients/Max_11_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╙
#gradients/Max_11_grad/DynamicStitchDynamicStitchgradients/Max_11_grad/rangegradients/Max_11_grad/modgradients/Max_11_grad/Shapegradients/Max_11_grad/Fill*
N*
T0*
_output_shapes
:
И
gradients/Max_11_grad/ReshapeReshapeMax_11#gradients/Max_11_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
▒
gradients/Max_11_grad/Reshape_1Reshape-gradients/sub_5_grad/tuple/control_dependency#gradients/Max_11_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Ц
gradients/Max_11_grad/EqualEqualgradients/Max_11_grad/ReshapeMax_11/input*
T0*
_output_shapes
:*
incompatible_shape_error(
Г
gradients/Max_11_grad/CastCastgradients/Max_11_grad/Equal*

DstT0*

SrcT0
*
Truncate( *
_output_shapes
:
Д
gradients/Max_11_grad/SumSumgradients/Max_11_grad/Castrange_13*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Э
gradients/Max_11_grad/Reshape_2Reshapegradients/Max_11_grad/Sum#gradients/Max_11_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
К
gradients/Max_11_grad/truedivRealDivgradients/Max_11_grad/Castgradients/Max_11_grad/Reshape_2*
T0*
_output_shapes
:
Е
gradients/Max_11_grad/mulMulgradients/Max_11_grad/truedivgradients/Max_11_grad/Reshape_1*
T0*
_output_shapes
:
e
gradients/Min_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
\
gradients/Min_11_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
m
gradients/Min_11_grad/addAddV2range_12gradients/Min_11_grad/Size*
T0*
_output_shapes
:
Б
gradients/Min_11_grad/modFloorModgradients/Min_11_grad/addgradients/Min_11_grad/Size*
T0*
_output_shapes
:
g
gradients/Min_11_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
c
!gradients/Min_11_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
c
!gradients/Min_11_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
о
gradients/Min_11_grad/rangeRange!gradients/Min_11_grad/range/startgradients/Min_11_grad/Size!gradients/Min_11_grad/range/delta*

Tidx0*
_output_shapes
:
b
 gradients/Min_11_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ъ
gradients/Min_11_grad/FillFillgradients/Min_11_grad/Shape_1 gradients/Min_11_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╙
#gradients/Min_11_grad/DynamicStitchDynamicStitchgradients/Min_11_grad/rangegradients/Min_11_grad/modgradients/Min_11_grad/Shapegradients/Min_11_grad/Fill*
N*
T0*
_output_shapes
:
И
gradients/Min_11_grad/ReshapeReshapeMin_11#gradients/Min_11_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
│
gradients/Min_11_grad/Reshape_1Reshape/gradients/sub_5_grad/tuple/control_dependency_1#gradients/Min_11_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Ц
gradients/Min_11_grad/EqualEqualgradients/Min_11_grad/ReshapeMin_11/input*
T0*
_output_shapes
:*
incompatible_shape_error(
Г
gradients/Min_11_grad/CastCastgradients/Min_11_grad/Equal*

DstT0*

SrcT0
*
Truncate( *
_output_shapes
:
Д
gradients/Min_11_grad/SumSumgradients/Min_11_grad/Castrange_12*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Э
gradients/Min_11_grad/Reshape_2Reshapegradients/Min_11_grad/Sum#gradients/Min_11_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
К
gradients/Min_11_grad/truedivRealDivgradients/Min_11_grad/Castgradients/Min_11_grad/Reshape_2*
T0*
_output_shapes
:
Е
gradients/Min_11_grad/mulMulgradients/Min_11_grad/truedivgradients/Min_11_grad/Reshape_1*
T0*
_output_shapes
:
│
+gradients/activation_4/Relu_1_grad/ReluGradReluGrad*gradients/reshape_2/Reshape_1_grad/Reshapeactivation_4/Relu_1*
T0*0
_output_shapes
:         А
ж
%gradients/conv2d_3/Conv2D_grad/ShapeNShapeNreshape_1/Reshapeconv2d_3/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::*
out_type0
■
2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_3/Conv2D_grad/ShapeNconv2d_3/Conv2D/ReadVariableOp)gradients/activation_4/Relu_grad/ReluGrad*
T0*0
_output_shapes
:         А*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
э
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterreshape_1/Reshape'gradients/conv2d_3/Conv2D_grad/ShapeN:1)gradients/activation_4/Relu_grad/ReluGrad*
T0*(
_output_shapes
:АА*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
в
/gradients/conv2d_3/Conv2D_grad/tuple/group_depsNoOp4^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter3^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput
л
7gradients/conv2d_3/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_3/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         А
з
9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_3/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:АА
~
"gradients/Max_1/input_grad/unstackUnpackgradients/Max_1_grad/mul*
T0*
_output_shapes
: *

axis *	
num
~
"gradients/Min_1/input_grad/unstackUnpackgradients/Min_1_grad/mul*
T0*
_output_shapes
: *

axis *	
num
~
"gradients/Max_3/input_grad/unstackUnpackgradients/Max_3_grad/mul*
T0*
_output_shapes
: *

axis *	
num
~
"gradients/Min_3/input_grad/unstackUnpackgradients/Min_3_grad/mul*
T0*
_output_shapes
: *

axis *	
num
~
"gradients/Max_5/input_grad/unstackUnpackgradients/Max_5_grad/mul*
T0*
_output_shapes
: *

axis *	
num
~
"gradients/Min_5/input_grad/unstackUnpackgradients/Min_5_grad/mul*
T0*
_output_shapes
: *

axis *	
num
~
"gradients/Max_7/input_grad/unstackUnpackgradients/Max_7_grad/mul*
T0*
_output_shapes
: *

axis *	
num
~
"gradients/Min_7/input_grad/unstackUnpackgradients/Min_7_grad/mul*
T0*
_output_shapes
: *

axis *	
num
~
"gradients/Max_9/input_grad/unstackUnpackgradients/Max_9_grad/mul*
T0*
_output_shapes
: *

axis *	
num
~
"gradients/Min_9/input_grad/unstackUnpackgradients/Min_9_grad/mul*
T0*
_output_shapes
: *

axis *	
num
А
#gradients/Max_11/input_grad/unstackUnpackgradients/Max_11_grad/mul*
T0*
_output_shapes
: *

axis *	
num
А
#gradients/Min_11/input_grad/unstackUnpackgradients/Min_11_grad/mul*
T0*
_output_shapes
: *

axis *	
num
е
'gradients/conv2d_3/Conv2D_1_grad/ShapeNShapeNreshape_1/reshape_1_requantizeconv2d_3/mul_2*
N*
T0* 
_output_shapes
::*
out_type0
Ї
4gradients/conv2d_3/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/conv2d_3/Conv2D_1_grad/ShapeNconv2d_3/mul_2+gradients/activation_4/Relu_1_grad/ReluGrad*
T0*0
_output_shapes
:         А*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
А
5gradients/conv2d_3/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterreshape_1/reshape_1_requantize)gradients/conv2d_3/Conv2D_1_grad/ShapeN:1+gradients/activation_4/Relu_1_grad/ReluGrad*
T0*(
_output_shapes
:АА*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
и
1gradients/conv2d_3/Conv2D_1_grad/tuple/group_depsNoOp6^gradients/conv2d_3/Conv2D_1_grad/Conv2DBackpropFilter5^gradients/conv2d_3/Conv2D_1_grad/Conv2DBackpropInput
│
9gradients/conv2d_3/Conv2D_1_grad/tuple/control_dependencyIdentity4gradients/conv2d_3/Conv2D_1_grad/Conv2DBackpropInput2^gradients/conv2d_3/Conv2D_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/conv2d_3/Conv2D_1_grad/Conv2DBackpropInput*0
_output_shapes
:         А
п
;gradients/conv2d_3/Conv2D_1_grad/tuple/control_dependency_1Identity5gradients/conv2d_3/Conv2D_1_grad/Conv2DBackpropFilter2^gradients/conv2d_3/Conv2D_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/conv2d_3/Conv2D_1_grad/Conv2DBackpropFilter*(
_output_shapes
:АА
w
&gradients/reshape_1/Reshape_grad/ShapeShapeactivation_3/Relu*
T0*
_output_shapes
:*
out_type0
▌
(gradients/reshape_1/Reshape_grad/ReshapeReshape7gradients/conv2d_3/Conv2D_grad/tuple/control_dependency&gradients/reshape_1/Reshape_grad/Shape*
T0*
Tshape0*0
_output_shapes
:         А
q
gradients/Max_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
Y
gradients/Max_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
f
gradients/Max_grad/addAddV2Const_5gradients/Max_grad/Size*
T0*
_output_shapes
:
x
gradients/Max_grad/modFloorModgradients/Max_grad/addgradients/Max_grad/Size*
T0*
_output_shapes
:
d
gradients/Max_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
`
gradients/Max_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
`
gradients/Max_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
в
gradients/Max_grad/rangeRangegradients/Max_grad/range/startgradients/Max_grad/Sizegradients/Max_grad/range/delta*

Tidx0*
_output_shapes
:
_
gradients/Max_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
С
gradients/Max_grad/FillFillgradients/Max_grad/Shape_1gradients/Max_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
─
 gradients/Max_grad/DynamicStitchDynamicStitchgradients/Max_grad/rangegradients/Max_grad/modgradients/Max_grad/Shapegradients/Max_grad/Fill*
N*
T0*
_output_shapes
:
Л
gradients/Max_grad/ReshapeReshapeMax gradients/Max_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
м
gradients/Max_grad/Reshape_1Reshape"gradients/Max_1/input_grad/unstack gradients/Max_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
ж
gradients/Max_grad/EqualEqualgradients/Max_grad/Reshapedifferentiable_round_1*
T0*&
_output_shapes
:*
incompatible_shape_error(
Й
gradients/Max_grad/CastCastgradients/Max_grad/Equal*

DstT0*

SrcT0
*
Truncate( *&
_output_shapes
:
}
gradients/Max_grad/SumSumgradients/Max_grad/CastConst_5*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
а
gradients/Max_grad/Reshape_2Reshapegradients/Max_grad/Sum gradients/Max_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
Н
gradients/Max_grad/truedivRealDivgradients/Max_grad/Castgradients/Max_grad/Reshape_2*
T0*&
_output_shapes
:
И
gradients/Max_grad/mulMulgradients/Max_grad/truedivgradients/Max_grad/Reshape_1*
T0*&
_output_shapes
:
q
gradients/Min_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
Y
gradients/Min_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
f
gradients/Min_grad/addAddV2Const_3gradients/Min_grad/Size*
T0*
_output_shapes
:
x
gradients/Min_grad/modFloorModgradients/Min_grad/addgradients/Min_grad/Size*
T0*
_output_shapes
:
d
gradients/Min_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
`
gradients/Min_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
`
gradients/Min_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
в
gradients/Min_grad/rangeRangegradients/Min_grad/range/startgradients/Min_grad/Sizegradients/Min_grad/range/delta*

Tidx0*
_output_shapes
:
_
gradients/Min_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
С
gradients/Min_grad/FillFillgradients/Min_grad/Shape_1gradients/Min_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
─
 gradients/Min_grad/DynamicStitchDynamicStitchgradients/Min_grad/rangegradients/Min_grad/modgradients/Min_grad/Shapegradients/Min_grad/Fill*
N*
T0*
_output_shapes
:
Л
gradients/Min_grad/ReshapeReshapeMin gradients/Min_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
м
gradients/Min_grad/Reshape_1Reshape"gradients/Min_1/input_grad/unstack gradients/Min_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
д
gradients/Min_grad/EqualEqualgradients/Min_grad/Reshapedifferentiable_round*
T0*&
_output_shapes
:*
incompatible_shape_error(
Й
gradients/Min_grad/CastCastgradients/Min_grad/Equal*

DstT0*

SrcT0
*
Truncate( *&
_output_shapes
:
}
gradients/Min_grad/SumSumgradients/Min_grad/CastConst_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
а
gradients/Min_grad/Reshape_2Reshapegradients/Min_grad/Sum gradients/Min_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
Н
gradients/Min_grad/truedivRealDivgradients/Min_grad/Castgradients/Min_grad/Reshape_2*
T0*&
_output_shapes
:
И
gradients/Min_grad/mulMulgradients/Min_grad/truedivgradients/Min_grad/Reshape_1*
T0*&
_output_shapes
:
s
gradients/Max_2_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
[
gradients/Max_2_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
k
gradients/Max_2_grad/addAddV2Const_10gradients/Max_2_grad/Size*
T0*
_output_shapes
:
~
gradients/Max_2_grad/modFloorModgradients/Max_2_grad/addgradients/Max_2_grad/Size*
T0*
_output_shapes
:
f
gradients/Max_2_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Max_2_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Max_2_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Max_2_grad/rangeRange gradients/Max_2_grad/range/startgradients/Max_2_grad/Size gradients/Max_2_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Max_2_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Max_2_grad/FillFillgradients/Max_2_grad/Shape_1gradients/Max_2_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Max_2_grad/DynamicStitchDynamicStitchgradients/Max_2_grad/rangegradients/Max_2_grad/modgradients/Max_2_grad/Shapegradients/Max_2_grad/Fill*
N*
T0*
_output_shapes
:
С
gradients/Max_2_grad/ReshapeReshapeMax_2"gradients/Max_2_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
░
gradients/Max_2_grad/Reshape_1Reshape"gradients/Max_3/input_grad/unstack"gradients/Max_2_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
к
gradients/Max_2_grad/EqualEqualgradients/Max_2_grad/Reshapedifferentiable_round_3*
T0*&
_output_shapes
:
*
incompatible_shape_error(
Н
gradients/Max_2_grad/CastCastgradients/Max_2_grad/Equal*

DstT0*

SrcT0
*
Truncate( *&
_output_shapes
:

В
gradients/Max_2_grad/SumSumgradients/Max_2_grad/CastConst_10*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
ж
gradients/Max_2_grad/Reshape_2Reshapegradients/Max_2_grad/Sum"gradients/Max_2_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
У
gradients/Max_2_grad/truedivRealDivgradients/Max_2_grad/Castgradients/Max_2_grad/Reshape_2*
T0*&
_output_shapes
:

О
gradients/Max_2_grad/mulMulgradients/Max_2_grad/truedivgradients/Max_2_grad/Reshape_1*
T0*&
_output_shapes
:

s
gradients/Min_2_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
[
gradients/Min_2_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
j
gradients/Min_2_grad/addAddV2Const_8gradients/Min_2_grad/Size*
T0*
_output_shapes
:
~
gradients/Min_2_grad/modFloorModgradients/Min_2_grad/addgradients/Min_2_grad/Size*
T0*
_output_shapes
:
f
gradients/Min_2_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Min_2_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Min_2_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Min_2_grad/rangeRange gradients/Min_2_grad/range/startgradients/Min_2_grad/Size gradients/Min_2_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Min_2_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Min_2_grad/FillFillgradients/Min_2_grad/Shape_1gradients/Min_2_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Min_2_grad/DynamicStitchDynamicStitchgradients/Min_2_grad/rangegradients/Min_2_grad/modgradients/Min_2_grad/Shapegradients/Min_2_grad/Fill*
N*
T0*
_output_shapes
:
С
gradients/Min_2_grad/ReshapeReshapeMin_2"gradients/Min_2_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
░
gradients/Min_2_grad/Reshape_1Reshape"gradients/Min_3/input_grad/unstack"gradients/Min_2_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
к
gradients/Min_2_grad/EqualEqualgradients/Min_2_grad/Reshapedifferentiable_round_2*
T0*&
_output_shapes
:
*
incompatible_shape_error(
Н
gradients/Min_2_grad/CastCastgradients/Min_2_grad/Equal*

DstT0*

SrcT0
*
Truncate( *&
_output_shapes
:

Б
gradients/Min_2_grad/SumSumgradients/Min_2_grad/CastConst_8*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
ж
gradients/Min_2_grad/Reshape_2Reshapegradients/Min_2_grad/Sum"gradients/Min_2_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
У
gradients/Min_2_grad/truedivRealDivgradients/Min_2_grad/Castgradients/Min_2_grad/Reshape_2*
T0*&
_output_shapes
:

О
gradients/Min_2_grad/mulMulgradients/Min_2_grad/truedivgradients/Min_2_grad/Reshape_1*
T0*&
_output_shapes
:

s
gradients/Max_4_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
[
gradients/Max_4_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
k
gradients/Max_4_grad/addAddV2Const_15gradients/Max_4_grad/Size*
T0*
_output_shapes
:
~
gradients/Max_4_grad/modFloorModgradients/Max_4_grad/addgradients/Max_4_grad/Size*
T0*
_output_shapes
:
f
gradients/Max_4_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Max_4_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Max_4_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Max_4_grad/rangeRange gradients/Max_4_grad/range/startgradients/Max_4_grad/Size gradients/Max_4_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Max_4_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Max_4_grad/FillFillgradients/Max_4_grad/Shape_1gradients/Max_4_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Max_4_grad/DynamicStitchDynamicStitchgradients/Max_4_grad/rangegradients/Max_4_grad/modgradients/Max_4_grad/Shapegradients/Max_4_grad/Fill*
N*
T0*
_output_shapes
:
С
gradients/Max_4_grad/ReshapeReshapeMax_4"gradients/Max_4_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
░
gradients/Max_4_grad/Reshape_1Reshape"gradients/Max_5/input_grad/unstack"gradients/Max_4_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
л
gradients/Max_4_grad/EqualEqualgradients/Max_4_grad/Reshapedifferentiable_round_5*
T0*'
_output_shapes
:1
А*
incompatible_shape_error(
О
gradients/Max_4_grad/CastCastgradients/Max_4_grad/Equal*

DstT0*

SrcT0
*
Truncate( *'
_output_shapes
:1
А
В
gradients/Max_4_grad/SumSumgradients/Max_4_grad/CastConst_15*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
ж
gradients/Max_4_grad/Reshape_2Reshapegradients/Max_4_grad/Sum"gradients/Max_4_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
Ф
gradients/Max_4_grad/truedivRealDivgradients/Max_4_grad/Castgradients/Max_4_grad/Reshape_2*
T0*'
_output_shapes
:1
А
П
gradients/Max_4_grad/mulMulgradients/Max_4_grad/truedivgradients/Max_4_grad/Reshape_1*
T0*'
_output_shapes
:1
А
s
gradients/Min_4_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
[
gradients/Min_4_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
k
gradients/Min_4_grad/addAddV2Const_13gradients/Min_4_grad/Size*
T0*
_output_shapes
:
~
gradients/Min_4_grad/modFloorModgradients/Min_4_grad/addgradients/Min_4_grad/Size*
T0*
_output_shapes
:
f
gradients/Min_4_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Min_4_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Min_4_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Min_4_grad/rangeRange gradients/Min_4_grad/range/startgradients/Min_4_grad/Size gradients/Min_4_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Min_4_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Min_4_grad/FillFillgradients/Min_4_grad/Shape_1gradients/Min_4_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Min_4_grad/DynamicStitchDynamicStitchgradients/Min_4_grad/rangegradients/Min_4_grad/modgradients/Min_4_grad/Shapegradients/Min_4_grad/Fill*
N*
T0*
_output_shapes
:
С
gradients/Min_4_grad/ReshapeReshapeMin_4"gradients/Min_4_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
░
gradients/Min_4_grad/Reshape_1Reshape"gradients/Min_5/input_grad/unstack"gradients/Min_4_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
л
gradients/Min_4_grad/EqualEqualgradients/Min_4_grad/Reshapedifferentiable_round_4*
T0*'
_output_shapes
:1
А*
incompatible_shape_error(
О
gradients/Min_4_grad/CastCastgradients/Min_4_grad/Equal*

DstT0*

SrcT0
*
Truncate( *'
_output_shapes
:1
А
В
gradients/Min_4_grad/SumSumgradients/Min_4_grad/CastConst_13*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
ж
gradients/Min_4_grad/Reshape_2Reshapegradients/Min_4_grad/Sum"gradients/Min_4_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
Ф
gradients/Min_4_grad/truedivRealDivgradients/Min_4_grad/Castgradients/Min_4_grad/Reshape_2*
T0*'
_output_shapes
:1
А
П
gradients/Min_4_grad/mulMulgradients/Min_4_grad/truedivgradients/Min_4_grad/Reshape_1*
T0*'
_output_shapes
:1
А
p
gradients/Max_6_grad/ShapeShapedifferentiable_round_7*
T0*
_output_shapes
:*
out_type0
[
gradients/Max_6_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
k
gradients/Max_6_grad/addAddV2Const_20gradients/Max_6_grad/Size*
T0*
_output_shapes
:
~
gradients/Max_6_grad/modFloorModgradients/Max_6_grad/addgradients/Max_6_grad/Size*
T0*
_output_shapes
:
f
gradients/Max_6_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Max_6_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Max_6_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Max_6_grad/rangeRange gradients/Max_6_grad/range/startgradients/Max_6_grad/Size gradients/Max_6_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Max_6_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Max_6_grad/FillFillgradients/Max_6_grad/Shape_1gradients/Max_6_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Max_6_grad/DynamicStitchDynamicStitchgradients/Max_6_grad/rangegradients/Max_6_grad/modgradients/Max_6_grad/Shapegradients/Max_6_grad/Fill*
N*
T0*
_output_shapes
:
╡
gradients/Max_6_grad/ReshapeReshapeMax_6"gradients/Max_6_grad/DynamicStitch*
T0*
Tshape0*J
_output_shapes8
6:4                                    
╘
gradients/Max_6_grad/Reshape_1Reshape"gradients/Max_7/input_grad/unstack"gradients/Max_6_grad/DynamicStitch*
T0*
Tshape0*J
_output_shapes8
6:4                                    
╞
gradients/Max_6_grad/EqualEqualgradients/Max_6_grad/Reshapedifferentiable_round_7*
T0*B
_output_shapes0
.:,                           А*
incompatible_shape_error(
й
gradients/Max_6_grad/CastCastgradients/Max_6_grad/Equal*

DstT0*

SrcT0
*
Truncate( *B
_output_shapes0
.:,                           А
В
gradients/Max_6_grad/SumSumgradients/Max_6_grad/CastConst_20*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
╩
gradients/Max_6_grad/Reshape_2Reshapegradients/Max_6_grad/Sum"gradients/Max_6_grad/DynamicStitch*
T0*
Tshape0*J
_output_shapes8
6:4                                    
п
gradients/Max_6_grad/truedivRealDivgradients/Max_6_grad/Castgradients/Max_6_grad/Reshape_2*
T0*B
_output_shapes0
.:,                           А
Ш
gradients/Max_6_grad/mulMulgradients/Max_6_grad/truedivgradients/Max_6_grad/Reshape_1*
T0*0
_output_shapes
:         А
p
gradients/Min_6_grad/ShapeShapedifferentiable_round_6*
T0*
_output_shapes
:*
out_type0
[
gradients/Min_6_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
k
gradients/Min_6_grad/addAddV2Const_18gradients/Min_6_grad/Size*
T0*
_output_shapes
:
~
gradients/Min_6_grad/modFloorModgradients/Min_6_grad/addgradients/Min_6_grad/Size*
T0*
_output_shapes
:
f
gradients/Min_6_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Min_6_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Min_6_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Min_6_grad/rangeRange gradients/Min_6_grad/range/startgradients/Min_6_grad/Size gradients/Min_6_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Min_6_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Min_6_grad/FillFillgradients/Min_6_grad/Shape_1gradients/Min_6_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Min_6_grad/DynamicStitchDynamicStitchgradients/Min_6_grad/rangegradients/Min_6_grad/modgradients/Min_6_grad/Shapegradients/Min_6_grad/Fill*
N*
T0*
_output_shapes
:
╡
gradients/Min_6_grad/ReshapeReshapeMin_6"gradients/Min_6_grad/DynamicStitch*
T0*
Tshape0*J
_output_shapes8
6:4                                    
╘
gradients/Min_6_grad/Reshape_1Reshape"gradients/Min_7/input_grad/unstack"gradients/Min_6_grad/DynamicStitch*
T0*
Tshape0*J
_output_shapes8
6:4                                    
╞
gradients/Min_6_grad/EqualEqualgradients/Min_6_grad/Reshapedifferentiable_round_6*
T0*B
_output_shapes0
.:,                           А*
incompatible_shape_error(
й
gradients/Min_6_grad/CastCastgradients/Min_6_grad/Equal*

DstT0*

SrcT0
*
Truncate( *B
_output_shapes0
.:,                           А
В
gradients/Min_6_grad/SumSumgradients/Min_6_grad/CastConst_18*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
╩
gradients/Min_6_grad/Reshape_2Reshapegradients/Min_6_grad/Sum"gradients/Min_6_grad/DynamicStitch*
T0*
Tshape0*J
_output_shapes8
6:4                                    
п
gradients/Min_6_grad/truedivRealDivgradients/Min_6_grad/Castgradients/Min_6_grad/Reshape_2*
T0*B
_output_shapes0
.:,                           А
Ш
gradients/Min_6_grad/mulMulgradients/Min_6_grad/truedivgradients/Min_6_grad/Reshape_1*
T0*0
_output_shapes
:         А
s
gradients/Max_8_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
[
gradients/Max_8_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
k
gradients/Max_8_grad/addAddV2Const_25gradients/Max_8_grad/Size*
T0*
_output_shapes
:
~
gradients/Max_8_grad/modFloorModgradients/Max_8_grad/addgradients/Max_8_grad/Size*
T0*
_output_shapes
:
f
gradients/Max_8_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Max_8_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Max_8_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Max_8_grad/rangeRange gradients/Max_8_grad/range/startgradients/Max_8_grad/Size gradients/Max_8_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Max_8_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Max_8_grad/FillFillgradients/Max_8_grad/Shape_1gradients/Max_8_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Max_8_grad/DynamicStitchDynamicStitchgradients/Max_8_grad/rangegradients/Max_8_grad/modgradients/Max_8_grad/Shapegradients/Max_8_grad/Fill*
N*
T0*
_output_shapes
:
С
gradients/Max_8_grad/ReshapeReshapeMax_8"gradients/Max_8_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
░
gradients/Max_8_grad/Reshape_1Reshape"gradients/Max_9/input_grad/unstack"gradients/Max_8_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
м
gradients/Max_8_grad/EqualEqualgradients/Max_8_grad/Reshapedifferentiable_round_9*
T0*(
_output_shapes
:АА*
incompatible_shape_error(
П
gradients/Max_8_grad/CastCastgradients/Max_8_grad/Equal*

DstT0*

SrcT0
*
Truncate( *(
_output_shapes
:АА
В
gradients/Max_8_grad/SumSumgradients/Max_8_grad/CastConst_25*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
ж
gradients/Max_8_grad/Reshape_2Reshapegradients/Max_8_grad/Sum"gradients/Max_8_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
Х
gradients/Max_8_grad/truedivRealDivgradients/Max_8_grad/Castgradients/Max_8_grad/Reshape_2*
T0*(
_output_shapes
:АА
Р
gradients/Max_8_grad/mulMulgradients/Max_8_grad/truedivgradients/Max_8_grad/Reshape_1*
T0*(
_output_shapes
:АА
s
gradients/Min_8_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
[
gradients/Min_8_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
k
gradients/Min_8_grad/addAddV2Const_23gradients/Min_8_grad/Size*
T0*
_output_shapes
:
~
gradients/Min_8_grad/modFloorModgradients/Min_8_grad/addgradients/Min_8_grad/Size*
T0*
_output_shapes
:
f
gradients/Min_8_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 gradients/Min_8_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Min_8_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
к
gradients/Min_8_grad/rangeRange gradients/Min_8_grad/range/startgradients/Min_8_grad/Size gradients/Min_8_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Min_8_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ч
gradients/Min_8_grad/FillFillgradients/Min_8_grad/Shape_1gradients/Min_8_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╬
"gradients/Min_8_grad/DynamicStitchDynamicStitchgradients/Min_8_grad/rangegradients/Min_8_grad/modgradients/Min_8_grad/Shapegradients/Min_8_grad/Fill*
N*
T0*
_output_shapes
:
С
gradients/Min_8_grad/ReshapeReshapeMin_8"gradients/Min_8_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
░
gradients/Min_8_grad/Reshape_1Reshape"gradients/Min_9/input_grad/unstack"gradients/Min_8_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
м
gradients/Min_8_grad/EqualEqualgradients/Min_8_grad/Reshapedifferentiable_round_8*
T0*(
_output_shapes
:АА*
incompatible_shape_error(
П
gradients/Min_8_grad/CastCastgradients/Min_8_grad/Equal*

DstT0*

SrcT0
*
Truncate( *(
_output_shapes
:АА
В
gradients/Min_8_grad/SumSumgradients/Min_8_grad/CastConst_23*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
ж
gradients/Min_8_grad/Reshape_2Reshapegradients/Min_8_grad/Sum"gradients/Min_8_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
Х
gradients/Min_8_grad/truedivRealDivgradients/Min_8_grad/Castgradients/Min_8_grad/Reshape_2*
T0*(
_output_shapes
:АА
Р
gradients/Min_8_grad/mulMulgradients/Min_8_grad/truedivgradients/Min_8_grad/Reshape_1*
T0*(
_output_shapes
:АА
t
gradients/Max_10_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
\
gradients/Max_10_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
m
gradients/Max_10_grad/addAddV2Const_30gradients/Max_10_grad/Size*
T0*
_output_shapes
:
Б
gradients/Max_10_grad/modFloorModgradients/Max_10_grad/addgradients/Max_10_grad/Size*
T0*
_output_shapes
:
g
gradients/Max_10_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
c
!gradients/Max_10_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
c
!gradients/Max_10_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
о
gradients/Max_10_grad/rangeRange!gradients/Max_10_grad/range/startgradients/Max_10_grad/Size!gradients/Max_10_grad/range/delta*

Tidx0*
_output_shapes
:
b
 gradients/Max_10_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ъ
gradients/Max_10_grad/FillFillgradients/Max_10_grad/Shape_1 gradients/Max_10_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╙
#gradients/Max_10_grad/DynamicStitchDynamicStitchgradients/Max_10_grad/rangegradients/Max_10_grad/modgradients/Max_10_grad/Shapegradients/Max_10_grad/Fill*
N*
T0*
_output_shapes
:
Ф
gradients/Max_10_grad/ReshapeReshapeMax_10#gradients/Max_10_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
│
gradients/Max_10_grad/Reshape_1Reshape#gradients/Max_11/input_grad/unstack#gradients/Max_10_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
о
gradients/Max_10_grad/EqualEqualgradients/Max_10_grad/Reshapedifferentiable_round_11*
T0*'
_output_shapes
:А
*
incompatible_shape_error(
Р
gradients/Max_10_grad/CastCastgradients/Max_10_grad/Equal*

DstT0*

SrcT0
*
Truncate( *'
_output_shapes
:А

Д
gradients/Max_10_grad/SumSumgradients/Max_10_grad/CastConst_30*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
й
gradients/Max_10_grad/Reshape_2Reshapegradients/Max_10_grad/Sum#gradients/Max_10_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
Ч
gradients/Max_10_grad/truedivRealDivgradients/Max_10_grad/Castgradients/Max_10_grad/Reshape_2*
T0*'
_output_shapes
:А

Т
gradients/Max_10_grad/mulMulgradients/Max_10_grad/truedivgradients/Max_10_grad/Reshape_1*
T0*'
_output_shapes
:А

t
gradients/Min_10_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
\
gradients/Min_10_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
m
gradients/Min_10_grad/addAddV2Const_28gradients/Min_10_grad/Size*
T0*
_output_shapes
:
Б
gradients/Min_10_grad/modFloorModgradients/Min_10_grad/addgradients/Min_10_grad/Size*
T0*
_output_shapes
:
g
gradients/Min_10_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
c
!gradients/Min_10_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
c
!gradients/Min_10_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
о
gradients/Min_10_grad/rangeRange!gradients/Min_10_grad/range/startgradients/Min_10_grad/Size!gradients/Min_10_grad/range/delta*

Tidx0*
_output_shapes
:
b
 gradients/Min_10_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ъ
gradients/Min_10_grad/FillFillgradients/Min_10_grad/Shape_1 gradients/Min_10_grad/Fill/value*
T0*
_output_shapes
:*

index_type0
╙
#gradients/Min_10_grad/DynamicStitchDynamicStitchgradients/Min_10_grad/rangegradients/Min_10_grad/modgradients/Min_10_grad/Shapegradients/Min_10_grad/Fill*
N*
T0*
_output_shapes
:
Ф
gradients/Min_10_grad/ReshapeReshapeMin_10#gradients/Min_10_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
│
gradients/Min_10_grad/Reshape_1Reshape#gradients/Min_11/input_grad/unstack#gradients/Min_10_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
о
gradients/Min_10_grad/EqualEqualgradients/Min_10_grad/Reshapedifferentiable_round_10*
T0*'
_output_shapes
:А
*
incompatible_shape_error(
Р
gradients/Min_10_grad/CastCastgradients/Min_10_grad/Equal*

DstT0*

SrcT0
*
Truncate( *'
_output_shapes
:А

Д
gradients/Min_10_grad/SumSumgradients/Min_10_grad/CastConst_28*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
й
gradients/Min_10_grad/Reshape_2Reshapegradients/Min_10_grad/Sum#gradients/Min_10_grad/DynamicStitch*
T0*
Tshape0*&
_output_shapes
:
Ч
gradients/Min_10_grad/truedivRealDivgradients/Min_10_grad/Castgradients/Min_10_grad/Reshape_2*
T0*'
_output_shapes
:А

Т
gradients/Min_10_grad/mulMulgradients/Min_10_grad/truedivgradients/Min_10_grad/Reshape_1*
T0*'
_output_shapes
:А

н
)gradients/activation_3/Relu_grad/ReluGradReluGrad(gradients/reshape_1/Reshape_grad/Reshapeactivation_3/Relu*
T0*0
_output_shapes
:         А
z
(gradients/reshape_1/Reshape_1_grad/ShapeShapeactivation_3/mul_2*
T0*
_output_shapes
:*
out_type0
у
*gradients/reshape_1/Reshape_1_grad/ReshapeReshape9gradients/conv2d_3/Conv2D_1_grad/tuple/control_dependency(gradients/reshape_1/Reshape_1_grad/Shape*
T0*
Tshape0*0
_output_shapes
:         А
p
-gradients/add_3_grad/BroadcastGradientArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
Ж
-gradients/add_3_grad/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*%
valueB"            
▐
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/add_3_grad/BroadcastGradientArgs/s0-gradients/add_3_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:         :         
Г
*gradients/add_3_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
б
gradients/add_3_grad/SumSumgradients/Max_grad/mul*gradients/add_3_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
"gradients/add_3_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ф
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sum"gradients/add_3_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
e
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/Max_grad/mul^gradients/add_3_grad/Reshape
╤
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape*
_output_shapes
: 
╫
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/Max_grad/mul&^gradients/add_3_grad/tuple/group_deps*
T0*)
_class
loc:@gradients/Max_grad/mul*&
_output_shapes
:
Г
*gradients/add_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
б
gradients/add_1_grad/SumSumgradients/Min_grad/mul*gradients/add_1_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
"gradients/add_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ф
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sum"gradients/add_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
e
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/Min_grad/mul^gradients/add_1_grad/Reshape
╤
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*
_output_shapes
: 
╫
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/Min_grad/mul&^gradients/add_1_grad/tuple/group_deps*
T0*)
_class
loc:@gradients/Min_grad/mul*&
_output_shapes
:
q
.gradients/add_10_grad/BroadcastGradientArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
З
.gradients/add_10_grad/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*%
valueB"         
   
с
+gradients/add_10_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/add_10_grad/BroadcastGradientArgs/s0.gradients/add_10_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:         :         
Д
+gradients/add_10_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
е
gradients/add_10_grad/SumSumgradients/Max_2_grad/mul+gradients/add_10_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/add_10_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/add_10_grad/ReshapeReshapegradients/add_10_grad/Sum#gradients/add_10_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
i
&gradients/add_10_grad/tuple/group_depsNoOp^gradients/Max_2_grad/mul^gradients/add_10_grad/Reshape
╒
.gradients/add_10_grad/tuple/control_dependencyIdentitygradients/add_10_grad/Reshape'^gradients/add_10_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_10_grad/Reshape*
_output_shapes
: 
▌
0gradients/add_10_grad/tuple/control_dependency_1Identitygradients/Max_2_grad/mul'^gradients/add_10_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Max_2_grad/mul*&
_output_shapes
:

Г
*gradients/add_8_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
г
gradients/add_8_grad/SumSumgradients/Min_2_grad/mul*gradients/add_8_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
"gradients/add_8_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ф
gradients/add_8_grad/ReshapeReshapegradients/add_8_grad/Sum"gradients/add_8_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
g
%gradients/add_8_grad/tuple/group_depsNoOp^gradients/Min_2_grad/mul^gradients/add_8_grad/Reshape
╤
-gradients/add_8_grad/tuple/control_dependencyIdentitygradients/add_8_grad/Reshape&^gradients/add_8_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_8_grad/Reshape*
_output_shapes
: 
█
/gradients/add_8_grad/tuple/control_dependency_1Identitygradients/Min_2_grad/mul&^gradients/add_8_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Min_2_grad/mul*&
_output_shapes
:

q
.gradients/add_17_grad/BroadcastGradientArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
З
.gradients/add_17_grad/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
с
+gradients/add_17_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/add_17_grad/BroadcastGradientArgs/s0.gradients/add_17_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:         :         
Д
+gradients/add_17_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
е
gradients/add_17_grad/SumSumgradients/Max_4_grad/mul+gradients/add_17_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/add_17_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/add_17_grad/ReshapeReshapegradients/add_17_grad/Sum#gradients/add_17_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
i
&gradients/add_17_grad/tuple/group_depsNoOp^gradients/Max_4_grad/mul^gradients/add_17_grad/Reshape
╒
.gradients/add_17_grad/tuple/control_dependencyIdentitygradients/add_17_grad/Reshape'^gradients/add_17_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_17_grad/Reshape*
_output_shapes
: 
▐
0gradients/add_17_grad/tuple/control_dependency_1Identitygradients/Max_4_grad/mul'^gradients/add_17_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Max_4_grad/mul*'
_output_shapes
:1
А
Д
+gradients/add_15_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
е
gradients/add_15_grad/SumSumgradients/Min_4_grad/mul+gradients/add_15_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/add_15_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/add_15_grad/ReshapeReshapegradients/add_15_grad/Sum#gradients/add_15_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
i
&gradients/add_15_grad/tuple/group_depsNoOp^gradients/Min_4_grad/mul^gradients/add_15_grad/Reshape
╒
.gradients/add_15_grad/tuple/control_dependencyIdentitygradients/add_15_grad/Reshape'^gradients/add_15_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_15_grad/Reshape*
_output_shapes
: 
▐
0gradients/add_15_grad/tuple/control_dependency_1Identitygradients/Min_4_grad/mul'^gradients/add_15_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Min_4_grad/mul*'
_output_shapes
:1
А
j
gradients/add_24_grad/ShapeShapeReadVariableOp_15*
T0*
_output_shapes
: *
out_type0
b
gradients/add_24_grad/Shape_1Shapemul_7*
T0*
_output_shapes
:*
out_type0
╜
+gradients/add_24_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_24_grad/Shapegradients/add_24_grad/Shape_1*
T0*2
_output_shapes 
:         :         
з
gradients/add_24_grad/SumSumgradients/Max_6_grad/mul+gradients/add_24_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
П
gradients/add_24_grad/ReshapeReshapegradients/add_24_grad/Sumgradients/add_24_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
л
gradients/add_24_grad/Sum_1Sumgradients/Max_6_grad/mul-gradients/add_24_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
п
gradients/add_24_grad/Reshape_1Reshapegradients/add_24_grad/Sum_1gradients/add_24_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:         А
p
&gradients/add_24_grad/tuple/group_depsNoOp^gradients/add_24_grad/Reshape ^gradients/add_24_grad/Reshape_1
╒
.gradients/add_24_grad/tuple/control_dependencyIdentitygradients/add_24_grad/Reshape'^gradients/add_24_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_24_grad/Reshape*
_output_shapes
: 
ї
0gradients/add_24_grad/tuple/control_dependency_1Identitygradients/add_24_grad/Reshape_1'^gradients/add_24_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_24_grad/Reshape_1*0
_output_shapes
:         А
j
gradients/add_22_grad/ShapeShapeReadVariableOp_13*
T0*
_output_shapes
: *
out_type0
b
gradients/add_22_grad/Shape_1Shapemul_6*
T0*
_output_shapes
:*
out_type0
╜
+gradients/add_22_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_22_grad/Shapegradients/add_22_grad/Shape_1*
T0*2
_output_shapes 
:         :         
з
gradients/add_22_grad/SumSumgradients/Min_6_grad/mul+gradients/add_22_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
П
gradients/add_22_grad/ReshapeReshapegradients/add_22_grad/Sumgradients/add_22_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
л
gradients/add_22_grad/Sum_1Sumgradients/Min_6_grad/mul-gradients/add_22_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
п
gradients/add_22_grad/Reshape_1Reshapegradients/add_22_grad/Sum_1gradients/add_22_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:         А
p
&gradients/add_22_grad/tuple/group_depsNoOp^gradients/add_22_grad/Reshape ^gradients/add_22_grad/Reshape_1
╒
.gradients/add_22_grad/tuple/control_dependencyIdentitygradients/add_22_grad/Reshape'^gradients/add_22_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_22_grad/Reshape*
_output_shapes
: 
ї
0gradients/add_22_grad/tuple/control_dependency_1Identitygradients/add_22_grad/Reshape_1'^gradients/add_22_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_22_grad/Reshape_1*0
_output_shapes
:         А
q
.gradients/add_31_grad/BroadcastGradientArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
З
.gradients/add_31_grad/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*%
valueB"А         А   
с
+gradients/add_31_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/add_31_grad/BroadcastGradientArgs/s0.gradients/add_31_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:         :         
Д
+gradients/add_31_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
е
gradients/add_31_grad/SumSumgradients/Max_8_grad/mul+gradients/add_31_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/add_31_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/add_31_grad/ReshapeReshapegradients/add_31_grad/Sum#gradients/add_31_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
i
&gradients/add_31_grad/tuple/group_depsNoOp^gradients/Max_8_grad/mul^gradients/add_31_grad/Reshape
╒
.gradients/add_31_grad/tuple/control_dependencyIdentitygradients/add_31_grad/Reshape'^gradients/add_31_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_31_grad/Reshape*
_output_shapes
: 
▀
0gradients/add_31_grad/tuple/control_dependency_1Identitygradients/Max_8_grad/mul'^gradients/add_31_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Max_8_grad/mul*(
_output_shapes
:АА
Д
+gradients/add_29_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
е
gradients/add_29_grad/SumSumgradients/Min_8_grad/mul+gradients/add_29_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/add_29_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/add_29_grad/ReshapeReshapegradients/add_29_grad/Sum#gradients/add_29_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
i
&gradients/add_29_grad/tuple/group_depsNoOp^gradients/Min_8_grad/mul^gradients/add_29_grad/Reshape
╒
.gradients/add_29_grad/tuple/control_dependencyIdentitygradients/add_29_grad/Reshape'^gradients/add_29_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_29_grad/Reshape*
_output_shapes
: 
▀
0gradients/add_29_grad/tuple/control_dependency_1Identitygradients/Min_8_grad/mul'^gradients/add_29_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Min_8_grad/mul*(
_output_shapes
:АА
q
.gradients/add_38_grad/BroadcastGradientArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
З
.gradients/add_38_grad/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*%
valueB"А         
   
с
+gradients/add_38_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/add_38_grad/BroadcastGradientArgs/s0.gradients/add_38_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:         :         
Д
+gradients/add_38_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
ж
gradients/add_38_grad/SumSumgradients/Max_10_grad/mul+gradients/add_38_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/add_38_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/add_38_grad/ReshapeReshapegradients/add_38_grad/Sum#gradients/add_38_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
j
&gradients/add_38_grad/tuple/group_depsNoOp^gradients/Max_10_grad/mul^gradients/add_38_grad/Reshape
╒
.gradients/add_38_grad/tuple/control_dependencyIdentitygradients/add_38_grad/Reshape'^gradients/add_38_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_38_grad/Reshape*
_output_shapes
: 
р
0gradients/add_38_grad/tuple/control_dependency_1Identitygradients/Max_10_grad/mul'^gradients/add_38_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Max_10_grad/mul*'
_output_shapes
:А

Д
+gradients/add_36_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
ж
gradients/add_36_grad/SumSumgradients/Min_10_grad/mul+gradients/add_36_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/add_36_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/add_36_grad/ReshapeReshapegradients/add_36_grad/Sum#gradients/add_36_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
j
&gradients/add_36_grad/tuple/group_depsNoOp^gradients/Min_10_grad/mul^gradients/add_36_grad/Reshape
╒
.gradients/add_36_grad/tuple/control_dependencyIdentitygradients/add_36_grad/Reshape'^gradients/add_36_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_36_grad/Reshape*
_output_shapes
: 
р
0gradients/add_36_grad/tuple/control_dependency_1Identitygradients/Min_10_grad/mul'^gradients/add_36_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Min_10_grad/mul*'
_output_shapes
:А

y
'gradients/activation_3/mul_2_grad/ShapeShapeactivation_3/Add_2*
T0*
_output_shapes
:*
out_type0
y
)gradients/activation_3/mul_2_grad/Shape_1Shapeactivation_3/pow*
T0*
_output_shapes
:*
out_type0
с
7gradients/activation_3/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/activation_3/mul_2_grad/Shape)gradients/activation_3/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
е
%gradients/activation_3/mul_2_grad/MulMul*gradients/reshape_1/Reshape_1_grad/Reshapeactivation_3/pow*
T0*0
_output_shapes
:         А
╠
%gradients/activation_3/mul_2_grad/SumSum%gradients/activation_3/mul_2_grad/Mul7gradients/activation_3/mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
═
)gradients/activation_3/mul_2_grad/ReshapeReshape%gradients/activation_3/mul_2_grad/Sum'gradients/activation_3/mul_2_grad/Shape*
T0*
Tshape0*0
_output_shapes
:         А
й
'gradients/activation_3/mul_2_grad/Mul_1Mulactivation_3/Add_2*gradients/reshape_1/Reshape_1_grad/Reshape*
T0*0
_output_shapes
:         А
╥
'gradients/activation_3/mul_2_grad/Sum_1Sum'gradients/activation_3/mul_2_grad/Mul_19gradients/activation_3/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
╙
+gradients/activation_3/mul_2_grad/Reshape_1Reshape'gradients/activation_3/mul_2_grad/Sum_1)gradients/activation_3/mul_2_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:         А
Ф
2gradients/activation_3/mul_2_grad/tuple/group_depsNoOp*^gradients/activation_3/mul_2_grad/Reshape,^gradients/activation_3/mul_2_grad/Reshape_1
Я
:gradients/activation_3/mul_2_grad/tuple/control_dependencyIdentity)gradients/activation_3/mul_2_grad/Reshape3^gradients/activation_3/mul_2_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/activation_3/mul_2_grad/Reshape*0
_output_shapes
:         А
е
<gradients/activation_3/mul_2_grad/tuple/control_dependency_1Identity+gradients/activation_3/mul_2_grad/Reshape_13^gradients/activation_3/mul_2_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/activation_3/mul_2_grad/Reshape_1*0
_output_shapes
:         А
М
gradients/mul_1_grad/MulMul/gradients/add_3_grad/tuple/control_dependency_1	truediv_1*
T0*&
_output_shapes
:
Г
*gradients/mul_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
г
gradients/mul_1_grad/SumSumgradients/mul_1_grad/Mul*gradients/mul_1_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
"gradients/mul_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ф
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sum"gradients/mul_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
Х
gradients/mul_1_grad/Mul_1MulReadVariableOp_2/gradients/add_3_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:
i
%gradients/mul_1_grad/tuple/group_depsNoOp^gradients/mul_1_grad/Mul_1^gradients/mul_1_grad/Reshape
╤
-gradients/mul_1_grad/tuple/control_dependencyIdentitygradients/mul_1_grad/Reshape&^gradients/mul_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_1_grad/Reshape*
_output_shapes
: 
▀
/gradients/mul_1_grad/tuple/control_dependency_1Identitygradients/mul_1_grad/Mul_1&^gradients/mul_1_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_1_grad/Mul_1*&
_output_shapes
:
И
gradients/mul_grad/MulMul/gradients/add_1_grad/tuple/control_dependency_1truediv*
T0*&
_output_shapes
:
Б
(gradients/mul_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
Э
gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
c
 gradients/mul_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
О
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sum gradients/mul_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
С
gradients/mul_grad/Mul_1MulReadVariableOp/gradients/add_1_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:
c
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Mul_1^gradients/mul_grad/Reshape
╔
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*
_output_shapes
: 
╫
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Mul_1$^gradients/mul_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_grad/Mul_1*&
_output_shapes
:
Н
gradients/mul_3_grad/MulMul0gradients/add_10_grad/tuple/control_dependency_1	truediv_4*
T0*&
_output_shapes
:

Г
*gradients/mul_3_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
г
gradients/mul_3_grad/SumSumgradients/mul_3_grad/Mul*gradients/mul_3_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
"gradients/mul_3_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ф
gradients/mul_3_grad/ReshapeReshapegradients/mul_3_grad/Sum"gradients/mul_3_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
Ц
gradients/mul_3_grad/Mul_1MulReadVariableOp_60gradients/add_10_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:

i
%gradients/mul_3_grad/tuple/group_depsNoOp^gradients/mul_3_grad/Mul_1^gradients/mul_3_grad/Reshape
╤
-gradients/mul_3_grad/tuple/control_dependencyIdentitygradients/mul_3_grad/Reshape&^gradients/mul_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_3_grad/Reshape*
_output_shapes
: 
▀
/gradients/mul_3_grad/tuple/control_dependency_1Identitygradients/mul_3_grad/Mul_1&^gradients/mul_3_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_3_grad/Mul_1*&
_output_shapes
:

М
gradients/mul_2_grad/MulMul/gradients/add_8_grad/tuple/control_dependency_1	truediv_3*
T0*&
_output_shapes
:

Г
*gradients/mul_2_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
г
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
"gradients/mul_2_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ф
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sum"gradients/mul_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
Х
gradients/mul_2_grad/Mul_1MulReadVariableOp_4/gradients/add_8_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:

i
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Mul_1^gradients/mul_2_grad/Reshape
╤
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape*
_output_shapes
: 
▀
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Mul_1&^gradients/mul_2_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_2_grad/Mul_1*&
_output_shapes
:

О
gradients/mul_5_grad/MulMul0gradients/add_17_grad/tuple/control_dependency_1	truediv_7*
T0*'
_output_shapes
:1
А
Г
*gradients/mul_5_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
г
gradients/mul_5_grad/SumSumgradients/mul_5_grad/Mul*gradients/mul_5_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
"gradients/mul_5_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ф
gradients/mul_5_grad/ReshapeReshapegradients/mul_5_grad/Sum"gradients/mul_5_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
Ш
gradients/mul_5_grad/Mul_1MulReadVariableOp_100gradients/add_17_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:1
А
i
%gradients/mul_5_grad/tuple/group_depsNoOp^gradients/mul_5_grad/Mul_1^gradients/mul_5_grad/Reshape
╤
-gradients/mul_5_grad/tuple/control_dependencyIdentitygradients/mul_5_grad/Reshape&^gradients/mul_5_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_5_grad/Reshape*
_output_shapes
: 
р
/gradients/mul_5_grad/tuple/control_dependency_1Identitygradients/mul_5_grad/Mul_1&^gradients/mul_5_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_5_grad/Mul_1*'
_output_shapes
:1
А
О
gradients/mul_4_grad/MulMul0gradients/add_15_grad/tuple/control_dependency_1	truediv_6*
T0*'
_output_shapes
:1
А
Г
*gradients/mul_4_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
г
gradients/mul_4_grad/SumSumgradients/mul_4_grad/Mul*gradients/mul_4_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
"gradients/mul_4_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ф
gradients/mul_4_grad/ReshapeReshapegradients/mul_4_grad/Sum"gradients/mul_4_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
Ч
gradients/mul_4_grad/Mul_1MulReadVariableOp_80gradients/add_15_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:1
А
i
%gradients/mul_4_grad/tuple/group_depsNoOp^gradients/mul_4_grad/Mul_1^gradients/mul_4_grad/Reshape
╤
-gradients/mul_4_grad/tuple/control_dependencyIdentitygradients/mul_4_grad/Reshape&^gradients/mul_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_4_grad/Reshape*
_output_shapes
: 
р
/gradients/mul_4_grad/tuple/control_dependency_1Identitygradients/mul_4_grad/Mul_1&^gradients/mul_4_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_4_grad/Mul_1*'
_output_shapes
:1
А
i
gradients/mul_7_grad/ShapeShapeReadVariableOp_14*
T0*
_output_shapes
: *
out_type0
f
gradients/mul_7_grad/Shape_1Shape
truediv_10*
T0*
_output_shapes
:*
out_type0
║
*gradients/mul_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_7_grad/Shapegradients/mul_7_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ш
gradients/mul_7_grad/MulMul0gradients/add_24_grad/tuple/control_dependency_1
truediv_10*
T0*0
_output_shapes
:         А
е
gradients/mul_7_grad/SumSumgradients/mul_7_grad/Mul*gradients/mul_7_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
М
gradients/mul_7_grad/ReshapeReshapegradients/mul_7_grad/Sumgradients/mul_7_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
б
gradients/mul_7_grad/Mul_1MulReadVariableOp_140gradients/add_24_grad/tuple/control_dependency_1*
T0*0
_output_shapes
:         А
л
gradients/mul_7_grad/Sum_1Sumgradients/mul_7_grad/Mul_1,gradients/mul_7_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
м
gradients/mul_7_grad/Reshape_1Reshapegradients/mul_7_grad/Sum_1gradients/mul_7_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:         А
m
%gradients/mul_7_grad/tuple/group_depsNoOp^gradients/mul_7_grad/Reshape^gradients/mul_7_grad/Reshape_1
╤
-gradients/mul_7_grad/tuple/control_dependencyIdentitygradients/mul_7_grad/Reshape&^gradients/mul_7_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_7_grad/Reshape*
_output_shapes
: 
ё
/gradients/mul_7_grad/tuple/control_dependency_1Identitygradients/mul_7_grad/Reshape_1&^gradients/mul_7_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_7_grad/Reshape_1*0
_output_shapes
:         А
i
gradients/mul_6_grad/ShapeShapeReadVariableOp_12*
T0*
_output_shapes
: *
out_type0
e
gradients/mul_6_grad/Shape_1Shape	truediv_9*
T0*
_output_shapes
:*
out_type0
║
*gradients/mul_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_6_grad/Shapegradients/mul_6_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ч
gradients/mul_6_grad/MulMul0gradients/add_22_grad/tuple/control_dependency_1	truediv_9*
T0*0
_output_shapes
:         А
е
gradients/mul_6_grad/SumSumgradients/mul_6_grad/Mul*gradients/mul_6_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
М
gradients/mul_6_grad/ReshapeReshapegradients/mul_6_grad/Sumgradients/mul_6_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
б
gradients/mul_6_grad/Mul_1MulReadVariableOp_120gradients/add_22_grad/tuple/control_dependency_1*
T0*0
_output_shapes
:         А
л
gradients/mul_6_grad/Sum_1Sumgradients/mul_6_grad/Mul_1,gradients/mul_6_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
м
gradients/mul_6_grad/Reshape_1Reshapegradients/mul_6_grad/Sum_1gradients/mul_6_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:         А
m
%gradients/mul_6_grad/tuple/group_depsNoOp^gradients/mul_6_grad/Reshape^gradients/mul_6_grad/Reshape_1
╤
-gradients/mul_6_grad/tuple/control_dependencyIdentitygradients/mul_6_grad/Reshape&^gradients/mul_6_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_6_grad/Reshape*
_output_shapes
: 
ё
/gradients/mul_6_grad/tuple/control_dependency_1Identitygradients/mul_6_grad/Reshape_1&^gradients/mul_6_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_6_grad/Reshape_1*0
_output_shapes
:         А
Р
gradients/mul_9_grad/MulMul0gradients/add_31_grad/tuple/control_dependency_1
truediv_13*
T0*(
_output_shapes
:АА
Г
*gradients/mul_9_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
г
gradients/mul_9_grad/SumSumgradients/mul_9_grad/Mul*gradients/mul_9_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
"gradients/mul_9_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ф
gradients/mul_9_grad/ReshapeReshapegradients/mul_9_grad/Sum"gradients/mul_9_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
Щ
gradients/mul_9_grad/Mul_1MulReadVariableOp_180gradients/add_31_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:АА
i
%gradients/mul_9_grad/tuple/group_depsNoOp^gradients/mul_9_grad/Mul_1^gradients/mul_9_grad/Reshape
╤
-gradients/mul_9_grad/tuple/control_dependencyIdentitygradients/mul_9_grad/Reshape&^gradients/mul_9_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_9_grad/Reshape*
_output_shapes
: 
с
/gradients/mul_9_grad/tuple/control_dependency_1Identitygradients/mul_9_grad/Mul_1&^gradients/mul_9_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_9_grad/Mul_1*(
_output_shapes
:АА
Р
gradients/mul_8_grad/MulMul0gradients/add_29_grad/tuple/control_dependency_1
truediv_12*
T0*(
_output_shapes
:АА
Г
*gradients/mul_8_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
г
gradients/mul_8_grad/SumSumgradients/mul_8_grad/Mul*gradients/mul_8_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
"gradients/mul_8_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ф
gradients/mul_8_grad/ReshapeReshapegradients/mul_8_grad/Sum"gradients/mul_8_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
Щ
gradients/mul_8_grad/Mul_1MulReadVariableOp_160gradients/add_29_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:АА
i
%gradients/mul_8_grad/tuple/group_depsNoOp^gradients/mul_8_grad/Mul_1^gradients/mul_8_grad/Reshape
╤
-gradients/mul_8_grad/tuple/control_dependencyIdentitygradients/mul_8_grad/Reshape&^gradients/mul_8_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_8_grad/Reshape*
_output_shapes
: 
с
/gradients/mul_8_grad/tuple/control_dependency_1Identitygradients/mul_8_grad/Mul_1&^gradients/mul_8_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_8_grad/Mul_1*(
_output_shapes
:АА
Р
gradients/mul_11_grad/MulMul0gradients/add_38_grad/tuple/control_dependency_1
truediv_16*
T0*'
_output_shapes
:А

Д
+gradients/mul_11_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
ж
gradients/mul_11_grad/SumSumgradients/mul_11_grad/Mul+gradients/mul_11_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/mul_11_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/mul_11_grad/ReshapeReshapegradients/mul_11_grad/Sum#gradients/mul_11_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
Щ
gradients/mul_11_grad/Mul_1MulReadVariableOp_220gradients/add_38_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:А

l
&gradients/mul_11_grad/tuple/group_depsNoOp^gradients/mul_11_grad/Mul_1^gradients/mul_11_grad/Reshape
╒
.gradients/mul_11_grad/tuple/control_dependencyIdentitygradients/mul_11_grad/Reshape'^gradients/mul_11_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/mul_11_grad/Reshape*
_output_shapes
: 
ф
0gradients/mul_11_grad/tuple/control_dependency_1Identitygradients/mul_11_grad/Mul_1'^gradients/mul_11_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_11_grad/Mul_1*'
_output_shapes
:А

Р
gradients/mul_10_grad/MulMul0gradients/add_36_grad/tuple/control_dependency_1
truediv_15*
T0*'
_output_shapes
:А

Д
+gradients/mul_10_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
ж
gradients/mul_10_grad/SumSumgradients/mul_10_grad/Mul+gradients/mul_10_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/mul_10_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/mul_10_grad/ReshapeReshapegradients/mul_10_grad/Sum#gradients/mul_10_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
Щ
gradients/mul_10_grad/Mul_1MulReadVariableOp_200gradients/add_36_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:А

l
&gradients/mul_10_grad/tuple/group_depsNoOp^gradients/mul_10_grad/Mul_1^gradients/mul_10_grad/Reshape
╒
.gradients/mul_10_grad/tuple/control_dependencyIdentitygradients/mul_10_grad/Reshape'^gradients/mul_10_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/mul_10_grad/Reshape*
_output_shapes
: 
ф
0gradients/mul_10_grad/tuple/control_dependency_1Identitygradients/mul_10_grad/Mul_1'^gradients/mul_10_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/mul_10_grad/Mul_1*'
_output_shapes
:А

~
'gradients/activation_3/Add_2_grad/ShapeShapeactivation_3/SelectV2_1*
T0*
_output_shapes
:*
out_type0
~
)gradients/activation_3/Add_2_grad/Shape_1Shapeactivation_3/SelectV2*
T0*
_output_shapes
:*
out_type0
с
7gradients/activation_3/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/activation_3/Add_2_grad/Shape)gradients/activation_3/Add_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
с
%gradients/activation_3/Add_2_grad/SumSum:gradients/activation_3/mul_2_grad/tuple/control_dependency7gradients/activation_3/Add_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
═
)gradients/activation_3/Add_2_grad/ReshapeReshape%gradients/activation_3/Add_2_grad/Sum'gradients/activation_3/Add_2_grad/Shape*
T0*
Tshape0*0
_output_shapes
:         А
х
'gradients/activation_3/Add_2_grad/Sum_1Sum:gradients/activation_3/mul_2_grad/tuple/control_dependency9gradients/activation_3/Add_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
╙
+gradients/activation_3/Add_2_grad/Reshape_1Reshape'gradients/activation_3/Add_2_grad/Sum_1)gradients/activation_3/Add_2_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:         А
Ф
2gradients/activation_3/Add_2_grad/tuple/group_depsNoOp*^gradients/activation_3/Add_2_grad/Reshape,^gradients/activation_3/Add_2_grad/Reshape_1
Я
:gradients/activation_3/Add_2_grad/tuple/control_dependencyIdentity)gradients/activation_3/Add_2_grad/Reshape3^gradients/activation_3/Add_2_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/activation_3/Add_2_grad/Reshape*0
_output_shapes
:         А
е
<gradients/activation_3/Add_2_grad/tuple/control_dependency_1Identity+gradients/activation_3/Add_2_grad/Reshape_13^gradients/activation_3/Add_2_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/activation_3/Add_2_grad/Reshape_1*0
_output_shapes
:         А
u
%gradients/activation_3/pow_grad/ShapeShapeactivation_3/pow/x*
T0*
_output_shapes
: *
out_type0
И
'gradients/activation_3/pow_grad/Shape_1Shape!activation_3/differentiable_round*
T0*
_output_shapes
:*
out_type0
█
5gradients/activation_3/pow_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/activation_3/pow_grad/Shape'gradients/activation_3/pow_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╞
#gradients/activation_3/pow_grad/mulMul<gradients/activation_3/mul_2_grad/tuple/control_dependency_1!activation_3/differentiable_round*
T0*0
_output_shapes
:         А
j
%gradients/activation_3/pow_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
п
#gradients/activation_3/pow_grad/subSub!activation_3/differentiable_round%gradients/activation_3/pow_grad/sub/y*
T0*0
_output_shapes
:         А
Ю
#gradients/activation_3/pow_grad/PowPowactivation_3/pow/x#gradients/activation_3/pow_grad/sub*
T0*0
_output_shapes
:         А
▒
%gradients/activation_3/pow_grad/mul_1Mul#gradients/activation_3/pow_grad/mul#gradients/activation_3/pow_grad/Pow*
T0*0
_output_shapes
:         А
╚
#gradients/activation_3/pow_grad/SumSum%gradients/activation_3/pow_grad/mul_15gradients/activation_3/pow_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
н
'gradients/activation_3/pow_grad/ReshapeReshape#gradients/activation_3/pow_grad/Sum%gradients/activation_3/pow_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
n
)gradients/activation_3/pow_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
Т
'gradients/activation_3/pow_grad/GreaterGreateractivation_3/pow/x)gradients/activation_3/pow_grad/Greater/y*
T0*
_output_shapes
: 
r
/gradients/activation_3/pow_grad/ones_like/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
t
/gradients/activation_3/pow_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
╞
)gradients/activation_3/pow_grad/ones_likeFill/gradients/activation_3/pow_grad/ones_like/Shape/gradients/activation_3/pow_grad/ones_like/Const*
T0*
_output_shapes
: *

index_type0
╣
&gradients/activation_3/pow_grad/SelectSelect'gradients/activation_3/pow_grad/Greateractivation_3/pow/x)gradients/activation_3/pow_grad/ones_like*
T0*
_output_shapes
: 
s
#gradients/activation_3/pow_grad/LogLog&gradients/activation_3/pow_grad/Select*
T0*
_output_shapes
: 
o
*gradients/activation_3/pow_grad/zeros_likeConst*
_output_shapes
: *
dtype0*
valueB
 *    
═
(gradients/activation_3/pow_grad/Select_1Select'gradients/activation_3/pow_grad/Greater#gradients/activation_3/pow_grad/Log*gradients/activation_3/pow_grad/zeros_like*
T0*
_output_shapes
: 
╖
%gradients/activation_3/pow_grad/mul_2Mul<gradients/activation_3/mul_2_grad/tuple/control_dependency_1activation_3/pow*
T0*0
_output_shapes
:         А
╕
%gradients/activation_3/pow_grad/mul_3Mul%gradients/activation_3/pow_grad/mul_2(gradients/activation_3/pow_grad/Select_1*
T0*0
_output_shapes
:         А
╠
%gradients/activation_3/pow_grad/Sum_1Sum%gradients/activation_3/pow_grad/mul_37gradients/activation_3/pow_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
═
)gradients/activation_3/pow_grad/Reshape_1Reshape%gradients/activation_3/pow_grad/Sum_1'gradients/activation_3/pow_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:         А
О
0gradients/activation_3/pow_grad/tuple/group_depsNoOp(^gradients/activation_3/pow_grad/Reshape*^gradients/activation_3/pow_grad/Reshape_1
¤
8gradients/activation_3/pow_grad/tuple/control_dependencyIdentity'gradients/activation_3/pow_grad/Reshape1^gradients/activation_3/pow_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/activation_3/pow_grad/Reshape*
_output_shapes
: 
Э
:gradients/activation_3/pow_grad/tuple/control_dependency_1Identity)gradients/activation_3/pow_grad/Reshape_11^gradients/activation_3/pow_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/activation_3/pow_grad/Reshape_1*0
_output_shapes
:         А
w
gradients/truediv_1_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
c
 gradients/truediv_1_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╞
.gradients/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_1_grad/Shape gradients/truediv_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ф
 gradients/truediv_1_grad/RealDivRealDiv/gradients/mul_1_grad/tuple/control_dependency_1Log_3*
T0*&
_output_shapes
:
┐
gradients/truediv_1_grad/SumSum gradients/truediv_1_grad/RealDiv.gradients/truediv_1_grad/BroadcastGradientArgs*
T0*

Tidx0*"
_output_shapes
:*
	keep_dims( 
и
 gradients/truediv_1_grad/ReshapeReshapegradients/truediv_1_grad/Sumgradients/truediv_1_grad/Shape*
T0*
Tshape0*&
_output_shapes
:
[
gradients/truediv_1_grad/NegNegLog_2*
T0*&
_output_shapes
:
Г
"gradients/truediv_1_grad/RealDiv_1RealDivgradients/truediv_1_grad/NegLog_3*
T0*&
_output_shapes
:
Й
"gradients/truediv_1_grad/RealDiv_2RealDiv"gradients/truediv_1_grad/RealDiv_1Log_3*
T0*&
_output_shapes
:
й
gradients/truediv_1_grad/mulMul/gradients/mul_1_grad/tuple/control_dependency_1"gradients/truediv_1_grad/RealDiv_2*
T0*&
_output_shapes
:
│
gradients/truediv_1_grad/Sum_1Sumgradients/truediv_1_grad/mul0gradients/truediv_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ю
"gradients/truediv_1_grad/Reshape_1Reshapegradients/truediv_1_grad/Sum_1 gradients/truediv_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
y
)gradients/truediv_1_grad/tuple/group_depsNoOp!^gradients/truediv_1_grad/Reshape#^gradients/truediv_1_grad/Reshape_1
ё
1gradients/truediv_1_grad/tuple/control_dependencyIdentity gradients/truediv_1_grad/Reshape*^gradients/truediv_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_1_grad/Reshape*&
_output_shapes
:
ч
3gradients/truediv_1_grad/tuple/control_dependency_1Identity"gradients/truediv_1_grad/Reshape_1*^gradients/truediv_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/truediv_1_grad/Reshape_1*
_output_shapes
: 
u
gradients/truediv_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
a
gradients/truediv_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
└
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Р
gradients/truediv_grad/RealDivRealDiv-gradients/mul_grad/tuple/control_dependency_1Log_1*
T0*&
_output_shapes
:
╣
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0*"
_output_shapes
:*
	keep_dims( 
в
gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*
T0*
Tshape0*&
_output_shapes
:
W
gradients/truediv_grad/NegNegLog*
T0*&
_output_shapes
:

 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/NegLog_1*
T0*&
_output_shapes
:
Е
 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1Log_1*
T0*&
_output_shapes
:
г
gradients/truediv_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1 gradients/truediv_grad/RealDiv_2*
T0*&
_output_shapes
:
н
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ш
 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
s
'gradients/truediv_grad/tuple/group_depsNoOp^gradients/truediv_grad/Reshape!^gradients/truediv_grad/Reshape_1
щ
/gradients/truediv_grad/tuple/control_dependencyIdentitygradients/truediv_grad/Reshape(^gradients/truediv_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/truediv_grad/Reshape*&
_output_shapes
:
▀
1gradients/truediv_grad/tuple/control_dependency_1Identity gradients/truediv_grad/Reshape_1(^gradients/truediv_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_grad/Reshape_1*
_output_shapes
: 
w
gradients/truediv_4_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
c
 gradients/truediv_4_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╞
.gradients/truediv_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_4_grad/Shape gradients/truediv_4_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ф
 gradients/truediv_4_grad/RealDivRealDiv/gradients/mul_3_grad/tuple/control_dependency_1Log_9*
T0*&
_output_shapes
:

├
gradients/truediv_4_grad/SumSum gradients/truediv_4_grad/RealDiv.gradients/truediv_4_grad/BroadcastGradientArgs*
T0*

Tidx0*&
_output_shapes
:
*
	keep_dims( 
и
 gradients/truediv_4_grad/ReshapeReshapegradients/truediv_4_grad/Sumgradients/truediv_4_grad/Shape*
T0*
Tshape0*&
_output_shapes
:

[
gradients/truediv_4_grad/NegNegLog_8*
T0*&
_output_shapes
:

Г
"gradients/truediv_4_grad/RealDiv_1RealDivgradients/truediv_4_grad/NegLog_9*
T0*&
_output_shapes
:

Й
"gradients/truediv_4_grad/RealDiv_2RealDiv"gradients/truediv_4_grad/RealDiv_1Log_9*
T0*&
_output_shapes
:

й
gradients/truediv_4_grad/mulMul/gradients/mul_3_grad/tuple/control_dependency_1"gradients/truediv_4_grad/RealDiv_2*
T0*&
_output_shapes
:

│
gradients/truediv_4_grad/Sum_1Sumgradients/truediv_4_grad/mul0gradients/truediv_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ю
"gradients/truediv_4_grad/Reshape_1Reshapegradients/truediv_4_grad/Sum_1 gradients/truediv_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
y
)gradients/truediv_4_grad/tuple/group_depsNoOp!^gradients/truediv_4_grad/Reshape#^gradients/truediv_4_grad/Reshape_1
ё
1gradients/truediv_4_grad/tuple/control_dependencyIdentity gradients/truediv_4_grad/Reshape*^gradients/truediv_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_4_grad/Reshape*&
_output_shapes
:

ч
3gradients/truediv_4_grad/tuple/control_dependency_1Identity"gradients/truediv_4_grad/Reshape_1*^gradients/truediv_4_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/truediv_4_grad/Reshape_1*
_output_shapes
: 
w
gradients/truediv_3_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
c
 gradients/truediv_3_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╞
.gradients/truediv_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_3_grad/Shape gradients/truediv_3_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ф
 gradients/truediv_3_grad/RealDivRealDiv/gradients/mul_2_grad/tuple/control_dependency_1Log_7*
T0*&
_output_shapes
:

├
gradients/truediv_3_grad/SumSum gradients/truediv_3_grad/RealDiv.gradients/truediv_3_grad/BroadcastGradientArgs*
T0*

Tidx0*&
_output_shapes
:
*
	keep_dims( 
и
 gradients/truediv_3_grad/ReshapeReshapegradients/truediv_3_grad/Sumgradients/truediv_3_grad/Shape*
T0*
Tshape0*&
_output_shapes
:

[
gradients/truediv_3_grad/NegNegLog_6*
T0*&
_output_shapes
:

Г
"gradients/truediv_3_grad/RealDiv_1RealDivgradients/truediv_3_grad/NegLog_7*
T0*&
_output_shapes
:

Й
"gradients/truediv_3_grad/RealDiv_2RealDiv"gradients/truediv_3_grad/RealDiv_1Log_7*
T0*&
_output_shapes
:

й
gradients/truediv_3_grad/mulMul/gradients/mul_2_grad/tuple/control_dependency_1"gradients/truediv_3_grad/RealDiv_2*
T0*&
_output_shapes
:

│
gradients/truediv_3_grad/Sum_1Sumgradients/truediv_3_grad/mul0gradients/truediv_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ю
"gradients/truediv_3_grad/Reshape_1Reshapegradients/truediv_3_grad/Sum_1 gradients/truediv_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
y
)gradients/truediv_3_grad/tuple/group_depsNoOp!^gradients/truediv_3_grad/Reshape#^gradients/truediv_3_grad/Reshape_1
ё
1gradients/truediv_3_grad/tuple/control_dependencyIdentity gradients/truediv_3_grad/Reshape*^gradients/truediv_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_3_grad/Reshape*&
_output_shapes
:

ч
3gradients/truediv_3_grad/tuple/control_dependency_1Identity"gradients/truediv_3_grad/Reshape_1*^gradients/truediv_3_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/truediv_3_grad/Reshape_1*
_output_shapes
: 
w
gradients/truediv_7_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
c
 gradients/truediv_7_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╞
.gradients/truediv_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_7_grad/Shape gradients/truediv_7_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ц
 gradients/truediv_7_grad/RealDivRealDiv/gradients/mul_5_grad/tuple/control_dependency_1Log_15*
T0*'
_output_shapes
:1
А
└
gradients/truediv_7_grad/SumSum gradients/truediv_7_grad/RealDiv.gradients/truediv_7_grad/BroadcastGradientArgs*
T0*

Tidx0*#
_output_shapes
:1
А*
	keep_dims( 
й
 gradients/truediv_7_grad/ReshapeReshapegradients/truediv_7_grad/Sumgradients/truediv_7_grad/Shape*
T0*
Tshape0*'
_output_shapes
:1
А
]
gradients/truediv_7_grad/NegNegLog_14*
T0*'
_output_shapes
:1
А
Е
"gradients/truediv_7_grad/RealDiv_1RealDivgradients/truediv_7_grad/NegLog_15*
T0*'
_output_shapes
:1
А
Л
"gradients/truediv_7_grad/RealDiv_2RealDiv"gradients/truediv_7_grad/RealDiv_1Log_15*
T0*'
_output_shapes
:1
А
к
gradients/truediv_7_grad/mulMul/gradients/mul_5_grad/tuple/control_dependency_1"gradients/truediv_7_grad/RealDiv_2*
T0*'
_output_shapes
:1
А
│
gradients/truediv_7_grad/Sum_1Sumgradients/truediv_7_grad/mul0gradients/truediv_7_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ю
"gradients/truediv_7_grad/Reshape_1Reshapegradients/truediv_7_grad/Sum_1 gradients/truediv_7_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
y
)gradients/truediv_7_grad/tuple/group_depsNoOp!^gradients/truediv_7_grad/Reshape#^gradients/truediv_7_grad/Reshape_1
Є
1gradients/truediv_7_grad/tuple/control_dependencyIdentity gradients/truediv_7_grad/Reshape*^gradients/truediv_7_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_7_grad/Reshape*'
_output_shapes
:1
А
ч
3gradients/truediv_7_grad/tuple/control_dependency_1Identity"gradients/truediv_7_grad/Reshape_1*^gradients/truediv_7_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/truediv_7_grad/Reshape_1*
_output_shapes
: 
w
gradients/truediv_6_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
c
 gradients/truediv_6_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╞
.gradients/truediv_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_6_grad/Shape gradients/truediv_6_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ц
 gradients/truediv_6_grad/RealDivRealDiv/gradients/mul_4_grad/tuple/control_dependency_1Log_13*
T0*'
_output_shapes
:1
А
└
gradients/truediv_6_grad/SumSum gradients/truediv_6_grad/RealDiv.gradients/truediv_6_grad/BroadcastGradientArgs*
T0*

Tidx0*#
_output_shapes
:1
А*
	keep_dims( 
й
 gradients/truediv_6_grad/ReshapeReshapegradients/truediv_6_grad/Sumgradients/truediv_6_grad/Shape*
T0*
Tshape0*'
_output_shapes
:1
А
]
gradients/truediv_6_grad/NegNegLog_12*
T0*'
_output_shapes
:1
А
Е
"gradients/truediv_6_grad/RealDiv_1RealDivgradients/truediv_6_grad/NegLog_13*
T0*'
_output_shapes
:1
А
Л
"gradients/truediv_6_grad/RealDiv_2RealDiv"gradients/truediv_6_grad/RealDiv_1Log_13*
T0*'
_output_shapes
:1
А
к
gradients/truediv_6_grad/mulMul/gradients/mul_4_grad/tuple/control_dependency_1"gradients/truediv_6_grad/RealDiv_2*
T0*'
_output_shapes
:1
А
│
gradients/truediv_6_grad/Sum_1Sumgradients/truediv_6_grad/mul0gradients/truediv_6_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ю
"gradients/truediv_6_grad/Reshape_1Reshapegradients/truediv_6_grad/Sum_1 gradients/truediv_6_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
y
)gradients/truediv_6_grad/tuple/group_depsNoOp!^gradients/truediv_6_grad/Reshape#^gradients/truediv_6_grad/Reshape_1
Є
1gradients/truediv_6_grad/tuple/control_dependencyIdentity gradients/truediv_6_grad/Reshape*^gradients/truediv_6_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_6_grad/Reshape*'
_output_shapes
:1
А
ч
3gradients/truediv_6_grad/tuple/control_dependency_1Identity"gradients/truediv_6_grad/Reshape_1*^gradients/truediv_6_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/truediv_6_grad/Reshape_1*
_output_shapes
: 
e
gradients/truediv_10_grad/ShapeShapeLog_20*
T0*
_output_shapes
:*
out_type0
d
!gradients/truediv_10_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╔
/gradients/truediv_10_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_10_grad/Shape!gradients/truediv_10_grad/Shape_1*
T0*2
_output_shapes 
:         :         
а
!gradients/truediv_10_grad/RealDivRealDiv/gradients/mul_7_grad/tuple/control_dependency_1Log_21*
T0*0
_output_shapes
:         А
╕
gradients/truediv_10_grad/SumSum!gradients/truediv_10_grad/RealDiv/gradients/truediv_10_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
╡
!gradients/truediv_10_grad/ReshapeReshapegradients/truediv_10_grad/Sumgradients/truediv_10_grad/Shape*
T0*
Tshape0*0
_output_shapes
:         А
g
gradients/truediv_10_grad/NegNegLog_20*
T0*0
_output_shapes
:         А
Р
#gradients/truediv_10_grad/RealDiv_1RealDivgradients/truediv_10_grad/NegLog_21*
T0*0
_output_shapes
:         А
Ц
#gradients/truediv_10_grad/RealDiv_2RealDiv#gradients/truediv_10_grad/RealDiv_1Log_21*
T0*0
_output_shapes
:         А
╡
gradients/truediv_10_grad/mulMul/gradients/mul_7_grad/tuple/control_dependency_1#gradients/truediv_10_grad/RealDiv_2*
T0*0
_output_shapes
:         А
╕
gradients/truediv_10_grad/Sum_1Sumgradients/truediv_10_grad/mul1gradients/truediv_10_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
б
#gradients/truediv_10_grad/Reshape_1Reshapegradients/truediv_10_grad/Sum_1!gradients/truediv_10_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
*gradients/truediv_10_grad/tuple/group_depsNoOp"^gradients/truediv_10_grad/Reshape$^gradients/truediv_10_grad/Reshape_1
 
2gradients/truediv_10_grad/tuple/control_dependencyIdentity!gradients/truediv_10_grad/Reshape+^gradients/truediv_10_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/truediv_10_grad/Reshape*0
_output_shapes
:         А
ы
4gradients/truediv_10_grad/tuple/control_dependency_1Identity#gradients/truediv_10_grad/Reshape_1+^gradients/truediv_10_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/truediv_10_grad/Reshape_1*
_output_shapes
: 
d
gradients/truediv_9_grad/ShapeShapeLog_18*
T0*
_output_shapes
:*
out_type0
c
 gradients/truediv_9_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╞
.gradients/truediv_9_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_9_grad/Shape gradients/truediv_9_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Я
 gradients/truediv_9_grad/RealDivRealDiv/gradients/mul_6_grad/tuple/control_dependency_1Log_19*
T0*0
_output_shapes
:         А
╡
gradients/truediv_9_grad/SumSum gradients/truediv_9_grad/RealDiv.gradients/truediv_9_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
▓
 gradients/truediv_9_grad/ReshapeReshapegradients/truediv_9_grad/Sumgradients/truediv_9_grad/Shape*
T0*
Tshape0*0
_output_shapes
:         А
f
gradients/truediv_9_grad/NegNegLog_18*
T0*0
_output_shapes
:         А
О
"gradients/truediv_9_grad/RealDiv_1RealDivgradients/truediv_9_grad/NegLog_19*
T0*0
_output_shapes
:         А
Ф
"gradients/truediv_9_grad/RealDiv_2RealDiv"gradients/truediv_9_grad/RealDiv_1Log_19*
T0*0
_output_shapes
:         А
│
gradients/truediv_9_grad/mulMul/gradients/mul_6_grad/tuple/control_dependency_1"gradients/truediv_9_grad/RealDiv_2*
T0*0
_output_shapes
:         А
╡
gradients/truediv_9_grad/Sum_1Sumgradients/truediv_9_grad/mul0gradients/truediv_9_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ю
"gradients/truediv_9_grad/Reshape_1Reshapegradients/truediv_9_grad/Sum_1 gradients/truediv_9_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
y
)gradients/truediv_9_grad/tuple/group_depsNoOp!^gradients/truediv_9_grad/Reshape#^gradients/truediv_9_grad/Reshape_1
√
1gradients/truediv_9_grad/tuple/control_dependencyIdentity gradients/truediv_9_grad/Reshape*^gradients/truediv_9_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_9_grad/Reshape*0
_output_shapes
:         А
ч
3gradients/truediv_9_grad/tuple/control_dependency_1Identity"gradients/truediv_9_grad/Reshape_1*^gradients/truediv_9_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/truediv_9_grad/Reshape_1*
_output_shapes
: 
x
gradients/truediv_13_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
d
!gradients/truediv_13_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╔
/gradients/truediv_13_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_13_grad/Shape!gradients/truediv_13_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ш
!gradients/truediv_13_grad/RealDivRealDiv/gradients/mul_9_grad/tuple/control_dependency_1Log_27*
T0*(
_output_shapes
:АА
└
gradients/truediv_13_grad/SumSum!gradients/truediv_13_grad/RealDiv/gradients/truediv_13_grad/BroadcastGradientArgs*
T0*

Tidx0* 
_output_shapes
:
АА*
	keep_dims( 
н
!gradients/truediv_13_grad/ReshapeReshapegradients/truediv_13_grad/Sumgradients/truediv_13_grad/Shape*
T0*
Tshape0*(
_output_shapes
:АА
_
gradients/truediv_13_grad/NegNegLog_26*
T0*(
_output_shapes
:АА
И
#gradients/truediv_13_grad/RealDiv_1RealDivgradients/truediv_13_grad/NegLog_27*
T0*(
_output_shapes
:АА
О
#gradients/truediv_13_grad/RealDiv_2RealDiv#gradients/truediv_13_grad/RealDiv_1Log_27*
T0*(
_output_shapes
:АА
н
gradients/truediv_13_grad/mulMul/gradients/mul_9_grad/tuple/control_dependency_1#gradients/truediv_13_grad/RealDiv_2*
T0*(
_output_shapes
:АА
╢
gradients/truediv_13_grad/Sum_1Sumgradients/truediv_13_grad/mul1gradients/truediv_13_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
б
#gradients/truediv_13_grad/Reshape_1Reshapegradients/truediv_13_grad/Sum_1!gradients/truediv_13_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
*gradients/truediv_13_grad/tuple/group_depsNoOp"^gradients/truediv_13_grad/Reshape$^gradients/truediv_13_grad/Reshape_1
ў
2gradients/truediv_13_grad/tuple/control_dependencyIdentity!gradients/truediv_13_grad/Reshape+^gradients/truediv_13_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/truediv_13_grad/Reshape*(
_output_shapes
:АА
ы
4gradients/truediv_13_grad/tuple/control_dependency_1Identity#gradients/truediv_13_grad/Reshape_1+^gradients/truediv_13_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/truediv_13_grad/Reshape_1*
_output_shapes
: 
x
gradients/truediv_12_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
d
!gradients/truediv_12_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╔
/gradients/truediv_12_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_12_grad/Shape!gradients/truediv_12_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ш
!gradients/truediv_12_grad/RealDivRealDiv/gradients/mul_8_grad/tuple/control_dependency_1Log_25*
T0*(
_output_shapes
:АА
└
gradients/truediv_12_grad/SumSum!gradients/truediv_12_grad/RealDiv/gradients/truediv_12_grad/BroadcastGradientArgs*
T0*

Tidx0* 
_output_shapes
:
АА*
	keep_dims( 
н
!gradients/truediv_12_grad/ReshapeReshapegradients/truediv_12_grad/Sumgradients/truediv_12_grad/Shape*
T0*
Tshape0*(
_output_shapes
:АА
_
gradients/truediv_12_grad/NegNegLog_24*
T0*(
_output_shapes
:АА
И
#gradients/truediv_12_grad/RealDiv_1RealDivgradients/truediv_12_grad/NegLog_25*
T0*(
_output_shapes
:АА
О
#gradients/truediv_12_grad/RealDiv_2RealDiv#gradients/truediv_12_grad/RealDiv_1Log_25*
T0*(
_output_shapes
:АА
н
gradients/truediv_12_grad/mulMul/gradients/mul_8_grad/tuple/control_dependency_1#gradients/truediv_12_grad/RealDiv_2*
T0*(
_output_shapes
:АА
╢
gradients/truediv_12_grad/Sum_1Sumgradients/truediv_12_grad/mul1gradients/truediv_12_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
б
#gradients/truediv_12_grad/Reshape_1Reshapegradients/truediv_12_grad/Sum_1!gradients/truediv_12_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
*gradients/truediv_12_grad/tuple/group_depsNoOp"^gradients/truediv_12_grad/Reshape$^gradients/truediv_12_grad/Reshape_1
ў
2gradients/truediv_12_grad/tuple/control_dependencyIdentity!gradients/truediv_12_grad/Reshape+^gradients/truediv_12_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/truediv_12_grad/Reshape*(
_output_shapes
:АА
ы
4gradients/truediv_12_grad/tuple/control_dependency_1Identity#gradients/truediv_12_grad/Reshape_1+^gradients/truediv_12_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/truediv_12_grad/Reshape_1*
_output_shapes
: 
x
gradients/truediv_16_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
d
!gradients/truediv_16_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╔
/gradients/truediv_16_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_16_grad/Shape!gradients/truediv_16_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ш
!gradients/truediv_16_grad/RealDivRealDiv0gradients/mul_11_grad/tuple/control_dependency_1Log_33*
T0*'
_output_shapes
:А

┐
gradients/truediv_16_grad/SumSum!gradients/truediv_16_grad/RealDiv/gradients/truediv_16_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:	А
*
	keep_dims( 
м
!gradients/truediv_16_grad/ReshapeReshapegradients/truediv_16_grad/Sumgradients/truediv_16_grad/Shape*
T0*
Tshape0*'
_output_shapes
:А

^
gradients/truediv_16_grad/NegNegLog_32*
T0*'
_output_shapes
:А

З
#gradients/truediv_16_grad/RealDiv_1RealDivgradients/truediv_16_grad/NegLog_33*
T0*'
_output_shapes
:А

Н
#gradients/truediv_16_grad/RealDiv_2RealDiv#gradients/truediv_16_grad/RealDiv_1Log_33*
T0*'
_output_shapes
:А

н
gradients/truediv_16_grad/mulMul0gradients/mul_11_grad/tuple/control_dependency_1#gradients/truediv_16_grad/RealDiv_2*
T0*'
_output_shapes
:А

╢
gradients/truediv_16_grad/Sum_1Sumgradients/truediv_16_grad/mul1gradients/truediv_16_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
б
#gradients/truediv_16_grad/Reshape_1Reshapegradients/truediv_16_grad/Sum_1!gradients/truediv_16_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
*gradients/truediv_16_grad/tuple/group_depsNoOp"^gradients/truediv_16_grad/Reshape$^gradients/truediv_16_grad/Reshape_1
Ў
2gradients/truediv_16_grad/tuple/control_dependencyIdentity!gradients/truediv_16_grad/Reshape+^gradients/truediv_16_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/truediv_16_grad/Reshape*'
_output_shapes
:А

ы
4gradients/truediv_16_grad/tuple/control_dependency_1Identity#gradients/truediv_16_grad/Reshape_1+^gradients/truediv_16_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/truediv_16_grad/Reshape_1*
_output_shapes
: 
x
gradients/truediv_15_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
d
!gradients/truediv_15_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╔
/gradients/truediv_15_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_15_grad/Shape!gradients/truediv_15_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ш
!gradients/truediv_15_grad/RealDivRealDiv0gradients/mul_10_grad/tuple/control_dependency_1Log_31*
T0*'
_output_shapes
:А

┐
gradients/truediv_15_grad/SumSum!gradients/truediv_15_grad/RealDiv/gradients/truediv_15_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:	А
*
	keep_dims( 
м
!gradients/truediv_15_grad/ReshapeReshapegradients/truediv_15_grad/Sumgradients/truediv_15_grad/Shape*
T0*
Tshape0*'
_output_shapes
:А

^
gradients/truediv_15_grad/NegNegLog_30*
T0*'
_output_shapes
:А

З
#gradients/truediv_15_grad/RealDiv_1RealDivgradients/truediv_15_grad/NegLog_31*
T0*'
_output_shapes
:А

Н
#gradients/truediv_15_grad/RealDiv_2RealDiv#gradients/truediv_15_grad/RealDiv_1Log_31*
T0*'
_output_shapes
:А

н
gradients/truediv_15_grad/mulMul0gradients/mul_10_grad/tuple/control_dependency_1#gradients/truediv_15_grad/RealDiv_2*
T0*'
_output_shapes
:А

╢
gradients/truediv_15_grad/Sum_1Sumgradients/truediv_15_grad/mul1gradients/truediv_15_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
б
#gradients/truediv_15_grad/Reshape_1Reshapegradients/truediv_15_grad/Sum_1!gradients/truediv_15_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
*gradients/truediv_15_grad/tuple/group_depsNoOp"^gradients/truediv_15_grad/Reshape$^gradients/truediv_15_grad/Reshape_1
Ў
2gradients/truediv_15_grad/tuple/control_dependencyIdentity!gradients/truediv_15_grad/Reshape+^gradients/truediv_15_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/truediv_15_grad/Reshape*'
_output_shapes
:А

ы
4gradients/truediv_15_grad/tuple/control_dependency_1Identity#gradients/truediv_15_grad/Reshape_1+^gradients/truediv_15_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/truediv_15_grad/Reshape_1*
_output_shapes
: 
q
,gradients/activation_3/SelectV2_1_grad/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    
°
/gradients/activation_3/SelectV2_1_grad/SelectV2SelectV2activation_3/LessEqual:gradients/activation_3/Add_2_grad/tuple/control_dependency,gradients/activation_3/SelectV2_1_grad/zeros*
T0*0
_output_shapes
:         А
~
,gradients/activation_3/SelectV2_1_grad/ShapeShapeactivation_3/mul_1*
T0*
_output_shapes
:*
out_type0
Е
.gradients/activation_3/SelectV2_1_grad/Shape_1Shapeactivation_3/SelectV2_1*
T0*
_output_shapes
:*
out_type0
Ё
<gradients/activation_3/SelectV2_1_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients/activation_3/SelectV2_1_grad/Shape.gradients/activation_3/SelectV2_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Т
*gradients/activation_3/SelectV2_1_grad/SumSum/gradients/activation_3/SelectV2_1_grad/SelectV2<gradients/activation_3/SelectV2_1_grad/BroadcastGradientArgs*
T0*

Tidx0*J
_output_shapes8
6:4                                    *
	keep_dims(
▄
.gradients/activation_3/SelectV2_1_grad/ReshapeReshape*gradients/activation_3/SelectV2_1_grad/Sum,gradients/activation_3/SelectV2_1_grad/Shape*
T0*
Tshape0*0
_output_shapes
:         А
·
1gradients/activation_3/SelectV2_1_grad/SelectV2_1SelectV2activation_3/LessEqual,gradients/activation_3/SelectV2_1_grad/zeros:gradients/activation_3/Add_2_grad/tuple/control_dependency*
T0*0
_output_shapes
:         А
З
.gradients/activation_3/SelectV2_1_grad/Shape_2Shapeactivation_3/zeros_like_1*
T0*
_output_shapes
:*
out_type0
Ї
>gradients/activation_3/SelectV2_1_grad/BroadcastGradientArgs_1BroadcastGradientArgs.gradients/activation_3/SelectV2_1_grad/Shape_2.gradients/activation_3/SelectV2_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ш
,gradients/activation_3/SelectV2_1_grad/Sum_1Sum1gradients/activation_3/SelectV2_1_grad/SelectV2_1>gradients/activation_3/SelectV2_1_grad/BroadcastGradientArgs_1*
T0*

Tidx0*J
_output_shapes8
6:4                                    *
	keep_dims(
т
0gradients/activation_3/SelectV2_1_grad/Reshape_1Reshape,gradients/activation_3/SelectV2_1_grad/Sum_1.gradients/activation_3/SelectV2_1_grad/Shape_2*
T0*
Tshape0*0
_output_shapes
:         А
г
7gradients/activation_3/SelectV2_1_grad/tuple/group_depsNoOp/^gradients/activation_3/SelectV2_1_grad/Reshape1^gradients/activation_3/SelectV2_1_grad/Reshape_1
│
?gradients/activation_3/SelectV2_1_grad/tuple/control_dependencyIdentity.gradients/activation_3/SelectV2_1_grad/Reshape8^gradients/activation_3/SelectV2_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/activation_3/SelectV2_1_grad/Reshape*0
_output_shapes
:         А
╣
Agradients/activation_3/SelectV2_1_grad/tuple/control_dependency_1Identity0gradients/activation_3/SelectV2_1_grad/Reshape_18^gradients/activation_3/SelectV2_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/activation_3/SelectV2_1_grad/Reshape_1*0
_output_shapes
:         А
o
*gradients/activation_3/SelectV2_grad/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    
∙
-gradients/activation_3/SelectV2_grad/SelectV2SelectV2activation_3/GreaterEqual<gradients/activation_3/Add_2_grad/tuple/control_dependency_1*gradients/activation_3/SelectV2_grad/zeros*
T0*0
_output_shapes
:         А
А
*gradients/activation_3/SelectV2_grad/ShapeShapeactivation_3/ones_like*
T0*
_output_shapes
:*
out_type0
Б
,gradients/activation_3/SelectV2_grad/Shape_1Shapeactivation_3/SelectV2*
T0*
_output_shapes
:*
out_type0
ъ
:gradients/activation_3/SelectV2_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/activation_3/SelectV2_grad/Shape,gradients/activation_3/SelectV2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
М
(gradients/activation_3/SelectV2_grad/SumSum-gradients/activation_3/SelectV2_grad/SelectV2:gradients/activation_3/SelectV2_grad/BroadcastGradientArgs*
T0*

Tidx0*J
_output_shapes8
6:4                                    *
	keep_dims(
╓
,gradients/activation_3/SelectV2_grad/ReshapeReshape(gradients/activation_3/SelectV2_grad/Sum*gradients/activation_3/SelectV2_grad/Shape*
T0*
Tshape0*0
_output_shapes
:         А
√
/gradients/activation_3/SelectV2_grad/SelectV2_1SelectV2activation_3/GreaterEqual*gradients/activation_3/SelectV2_grad/zeros<gradients/activation_3/Add_2_grad/tuple/control_dependency_1*
T0*0
_output_shapes
:         А
Г
,gradients/activation_3/SelectV2_grad/Shape_2Shapeactivation_3/zeros_like*
T0*
_output_shapes
:*
out_type0
ю
<gradients/activation_3/SelectV2_grad/BroadcastGradientArgs_1BroadcastGradientArgs,gradients/activation_3/SelectV2_grad/Shape_2,gradients/activation_3/SelectV2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Т
*gradients/activation_3/SelectV2_grad/Sum_1Sum/gradients/activation_3/SelectV2_grad/SelectV2_1<gradients/activation_3/SelectV2_grad/BroadcastGradientArgs_1*
T0*

Tidx0*J
_output_shapes8
6:4                                    *
	keep_dims(
▄
.gradients/activation_3/SelectV2_grad/Reshape_1Reshape*gradients/activation_3/SelectV2_grad/Sum_1,gradients/activation_3/SelectV2_grad/Shape_2*
T0*
Tshape0*0
_output_shapes
:         А
Э
5gradients/activation_3/SelectV2_grad/tuple/group_depsNoOp-^gradients/activation_3/SelectV2_grad/Reshape/^gradients/activation_3/SelectV2_grad/Reshape_1
л
=gradients/activation_3/SelectV2_grad/tuple/control_dependencyIdentity,gradients/activation_3/SelectV2_grad/Reshape6^gradients/activation_3/SelectV2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/activation_3/SelectV2_grad/Reshape*0
_output_shapes
:         А
▒
?gradients/activation_3/SelectV2_grad/tuple/control_dependency_1Identity.gradients/activation_3/SelectV2_grad/Reshape_16^gradients/activation_3/SelectV2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/activation_3/SelectV2_grad/Reshape_1*0
_output_shapes
:         А
Щ
gradients/Log_2_grad/Reciprocal
Reciprocaladd_22^gradients/truediv_1_grad/tuple/control_dependency*
T0*&
_output_shapes
:
д
gradients/Log_2_grad/mulMul1gradients/truediv_1_grad/tuple/control_dependencygradients/Log_2_grad/Reciprocal*
T0*&
_output_shapes
:
У
gradients/Log_grad/Reciprocal
Reciprocaladd0^gradients/truediv_grad/tuple/control_dependency*
T0*&
_output_shapes
:
Ю
gradients/Log_grad/mulMul/gradients/truediv_grad/tuple/control_dependencygradients/Log_grad/Reciprocal*
T0*&
_output_shapes
:
Щ
gradients/Log_8_grad/Reciprocal
Reciprocaladd_92^gradients/truediv_4_grad/tuple/control_dependency*
T0*&
_output_shapes
:

д
gradients/Log_8_grad/mulMul1gradients/truediv_4_grad/tuple/control_dependencygradients/Log_8_grad/Reciprocal*
T0*&
_output_shapes
:

Щ
gradients/Log_6_grad/Reciprocal
Reciprocaladd_72^gradients/truediv_3_grad/tuple/control_dependency*
T0*&
_output_shapes
:

д
gradients/Log_6_grad/mulMul1gradients/truediv_3_grad/tuple/control_dependencygradients/Log_6_grad/Reciprocal*
T0*&
_output_shapes
:

Ь
 gradients/Log_14_grad/Reciprocal
Reciprocaladd_162^gradients/truediv_7_grad/tuple/control_dependency*
T0*'
_output_shapes
:1
А
з
gradients/Log_14_grad/mulMul1gradients/truediv_7_grad/tuple/control_dependency gradients/Log_14_grad/Reciprocal*
T0*'
_output_shapes
:1
А
Ь
 gradients/Log_12_grad/Reciprocal
Reciprocaladd_142^gradients/truediv_6_grad/tuple/control_dependency*
T0*'
_output_shapes
:1
А
з
gradients/Log_12_grad/mulMul1gradients/truediv_6_grad/tuple/control_dependency gradients/Log_12_grad/Reciprocal*
T0*'
_output_shapes
:1
А
ж
 gradients/Log_20_grad/Reciprocal
Reciprocaladd_233^gradients/truediv_10_grad/tuple/control_dependency*
T0*0
_output_shapes
:         А
▒
gradients/Log_20_grad/mulMul2gradients/truediv_10_grad/tuple/control_dependency gradients/Log_20_grad/Reciprocal*
T0*0
_output_shapes
:         А
е
 gradients/Log_18_grad/Reciprocal
Reciprocaladd_212^gradients/truediv_9_grad/tuple/control_dependency*
T0*0
_output_shapes
:         А
░
gradients/Log_18_grad/mulMul1gradients/truediv_9_grad/tuple/control_dependency gradients/Log_18_grad/Reciprocal*
T0*0
_output_shapes
:         А
Ю
 gradients/Log_26_grad/Reciprocal
Reciprocaladd_303^gradients/truediv_13_grad/tuple/control_dependency*
T0*(
_output_shapes
:АА
й
gradients/Log_26_grad/mulMul2gradients/truediv_13_grad/tuple/control_dependency gradients/Log_26_grad/Reciprocal*
T0*(
_output_shapes
:АА
Ю
 gradients/Log_24_grad/Reciprocal
Reciprocaladd_283^gradients/truediv_12_grad/tuple/control_dependency*
T0*(
_output_shapes
:АА
й
gradients/Log_24_grad/mulMul2gradients/truediv_12_grad/tuple/control_dependency gradients/Log_24_grad/Reciprocal*
T0*(
_output_shapes
:АА
Э
 gradients/Log_32_grad/Reciprocal
Reciprocaladd_373^gradients/truediv_16_grad/tuple/control_dependency*
T0*'
_output_shapes
:А

и
gradients/Log_32_grad/mulMul2gradients/truediv_16_grad/tuple/control_dependency gradients/Log_32_grad/Reciprocal*
T0*'
_output_shapes
:А

Э
 gradients/Log_30_grad/Reciprocal
Reciprocaladd_353^gradients/truediv_15_grad/tuple/control_dependency*
T0*'
_output_shapes
:А

и
gradients/Log_30_grad/mulMul2gradients/truediv_15_grad/tuple/control_dependency gradients/Log_30_grad/Reciprocal*
T0*'
_output_shapes
:А

В
'gradients/activation_3/add_1_grad/ShapeShapeactivation_3/ReadVariableOp_1*
T0*
_output_shapes
: *
out_type0
y
)gradients/activation_3/add_1_grad/Shape_1Shapeactivation_3/mul*
T0*
_output_shapes
:*
out_type0
с
7gradients/activation_3/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/activation_3/add_1_grad/Shape)gradients/activation_3/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
с
%gradients/activation_3/add_1_grad/SumSum:gradients/activation_3/pow_grad/tuple/control_dependency_17gradients/activation_3/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
│
)gradients/activation_3/add_1_grad/ReshapeReshape%gradients/activation_3/add_1_grad/Sum'gradients/activation_3/add_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
х
'gradients/activation_3/add_1_grad/Sum_1Sum:gradients/activation_3/pow_grad/tuple/control_dependency_19gradients/activation_3/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
╙
+gradients/activation_3/add_1_grad/Reshape_1Reshape'gradients/activation_3/add_1_grad/Sum_1)gradients/activation_3/add_1_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:         А
Ф
2gradients/activation_3/add_1_grad/tuple/group_depsNoOp*^gradients/activation_3/add_1_grad/Reshape,^gradients/activation_3/add_1_grad/Reshape_1
Е
:gradients/activation_3/add_1_grad/tuple/control_dependencyIdentity)gradients/activation_3/add_1_grad/Reshape3^gradients/activation_3/add_1_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/activation_3/add_1_grad/Reshape*
_output_shapes
: 
е
<gradients/activation_3/add_1_grad/tuple/control_dependency_1Identity+gradients/activation_3/add_1_grad/Reshape_13^gradients/activation_3/add_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/activation_3/add_1_grad/Reshape_1*0
_output_shapes
:         А
Ж
-gradients/add_2_grad/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*%
valueB"            
p
-gradients/add_2_grad/BroadcastGradientArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 
▐
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/add_2_grad/BroadcastGradientArgs/s0-gradients/add_2_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:         :         
Г
*gradients/add_2_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
г
gradients/add_2_grad/SumSumgradients/Log_2_grad/mul*gradients/add_2_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
"gradients/add_2_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ф
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sum"gradients/add_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
g
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/Log_2_grad/mul^gradients/add_2_grad/Reshape
┘
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/Log_2_grad/mul&^gradients/add_2_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Log_2_grad/mul*&
_output_shapes
:
╙
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape*
_output_shapes
: 
Б
(gradients/add_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
Э
gradients/add_grad/SumSumgradients/Log_grad/mul(gradients/add_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
c
 gradients/add_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
О
gradients/add_grad/ReshapeReshapegradients/add_grad/Sum gradients/add_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
a
#gradients/add_grad/tuple/group_depsNoOp^gradients/Log_grad/mul^gradients/add_grad/Reshape
╤
+gradients/add_grad/tuple/control_dependencyIdentitygradients/Log_grad/mul$^gradients/add_grad/tuple/group_deps*
T0*)
_class
loc:@gradients/Log_grad/mul*&
_output_shapes
:
╦
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*
_output_shapes
: 
Ж
-gradients/add_9_grad/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*%
valueB"         
   
p
-gradients/add_9_grad/BroadcastGradientArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 
▐
*gradients/add_9_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/add_9_grad/BroadcastGradientArgs/s0-gradients/add_9_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:         :         
Г
*gradients/add_9_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
г
gradients/add_9_grad/SumSumgradients/Log_8_grad/mul*gradients/add_9_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
"gradients/add_9_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ф
gradients/add_9_grad/ReshapeReshapegradients/add_9_grad/Sum"gradients/add_9_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
g
%gradients/add_9_grad/tuple/group_depsNoOp^gradients/Log_8_grad/mul^gradients/add_9_grad/Reshape
┘
-gradients/add_9_grad/tuple/control_dependencyIdentitygradients/Log_8_grad/mul&^gradients/add_9_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Log_8_grad/mul*&
_output_shapes
:

╙
/gradients/add_9_grad/tuple/control_dependency_1Identitygradients/add_9_grad/Reshape&^gradients/add_9_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_9_grad/Reshape*
_output_shapes
: 
Г
*gradients/add_7_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
г
gradients/add_7_grad/SumSumgradients/Log_6_grad/mul*gradients/add_7_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
"gradients/add_7_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ф
gradients/add_7_grad/ReshapeReshapegradients/add_7_grad/Sum"gradients/add_7_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
g
%gradients/add_7_grad/tuple/group_depsNoOp^gradients/Log_6_grad/mul^gradients/add_7_grad/Reshape
┘
-gradients/add_7_grad/tuple/control_dependencyIdentitygradients/Log_6_grad/mul&^gradients/add_7_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/Log_6_grad/mul*&
_output_shapes
:

╙
/gradients/add_7_grad/tuple/control_dependency_1Identitygradients/add_7_grad/Reshape&^gradients/add_7_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_7_grad/Reshape*
_output_shapes
: 
З
.gradients/add_16_grad/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
q
.gradients/add_16_grad/BroadcastGradientArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 
с
+gradients/add_16_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/add_16_grad/BroadcastGradientArgs/s0.gradients/add_16_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:         :         
Д
+gradients/add_16_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
ж
gradients/add_16_grad/SumSumgradients/Log_14_grad/mul+gradients/add_16_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/add_16_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/add_16_grad/ReshapeReshapegradients/add_16_grad/Sum#gradients/add_16_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
j
&gradients/add_16_grad/tuple/group_depsNoOp^gradients/Log_14_grad/mul^gradients/add_16_grad/Reshape
▐
.gradients/add_16_grad/tuple/control_dependencyIdentitygradients/Log_14_grad/mul'^gradients/add_16_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_14_grad/mul*'
_output_shapes
:1
А
╫
0gradients/add_16_grad/tuple/control_dependency_1Identitygradients/add_16_grad/Reshape'^gradients/add_16_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_16_grad/Reshape*
_output_shapes
: 
Д
+gradients/add_14_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
ж
gradients/add_14_grad/SumSumgradients/Log_12_grad/mul+gradients/add_14_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/add_14_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/add_14_grad/ReshapeReshapegradients/add_14_grad/Sum#gradients/add_14_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
j
&gradients/add_14_grad/tuple/group_depsNoOp^gradients/Log_12_grad/mul^gradients/add_14_grad/Reshape
▐
.gradients/add_14_grad/tuple/control_dependencyIdentitygradients/Log_12_grad/mul'^gradients/add_14_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_12_grad/mul*'
_output_shapes
:1
А
╫
0gradients/add_14_grad/tuple/control_dependency_1Identitygradients/add_14_grad/Reshape'^gradients/add_14_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_14_grad/Reshape*
_output_shapes
: 
a
gradients/add_23_grad/ShapeShapeAbs_10*
T0*
_output_shapes
:*
out_type0
c
gradients/add_23_grad/Shape_1Shapeadd_23/y*
T0*
_output_shapes
: *
out_type0
╜
+gradients/add_23_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_23_grad/Shapegradients/add_23_grad/Shape_1*
T0*2
_output_shapes 
:         :         
и
gradients/add_23_grad/SumSumgradients/Log_20_grad/mul+gradients/add_23_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
й
gradients/add_23_grad/ReshapeReshapegradients/add_23_grad/Sumgradients/add_23_grad/Shape*
T0*
Tshape0*0
_output_shapes
:         А
м
gradients/add_23_grad/Sum_1Sumgradients/Log_20_grad/mul-gradients/add_23_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Х
gradients/add_23_grad/Reshape_1Reshapegradients/add_23_grad/Sum_1gradients/add_23_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
p
&gradients/add_23_grad/tuple/group_depsNoOp^gradients/add_23_grad/Reshape ^gradients/add_23_grad/Reshape_1
я
.gradients/add_23_grad/tuple/control_dependencyIdentitygradients/add_23_grad/Reshape'^gradients/add_23_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_23_grad/Reshape*0
_output_shapes
:         А
█
0gradients/add_23_grad/tuple/control_dependency_1Identitygradients/add_23_grad/Reshape_1'^gradients/add_23_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_23_grad/Reshape_1*
_output_shapes
: 
`
gradients/add_21_grad/ShapeShapeAbs_9*
T0*
_output_shapes
:*
out_type0
c
gradients/add_21_grad/Shape_1Shapeadd_21/y*
T0*
_output_shapes
: *
out_type0
╜
+gradients/add_21_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_21_grad/Shapegradients/add_21_grad/Shape_1*
T0*2
_output_shapes 
:         :         
и
gradients/add_21_grad/SumSumgradients/Log_18_grad/mul+gradients/add_21_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
й
gradients/add_21_grad/ReshapeReshapegradients/add_21_grad/Sumgradients/add_21_grad/Shape*
T0*
Tshape0*0
_output_shapes
:         А
м
gradients/add_21_grad/Sum_1Sumgradients/Log_18_grad/mul-gradients/add_21_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Х
gradients/add_21_grad/Reshape_1Reshapegradients/add_21_grad/Sum_1gradients/add_21_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
p
&gradients/add_21_grad/tuple/group_depsNoOp^gradients/add_21_grad/Reshape ^gradients/add_21_grad/Reshape_1
я
.gradients/add_21_grad/tuple/control_dependencyIdentitygradients/add_21_grad/Reshape'^gradients/add_21_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_21_grad/Reshape*0
_output_shapes
:         А
█
0gradients/add_21_grad/tuple/control_dependency_1Identitygradients/add_21_grad/Reshape_1'^gradients/add_21_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_21_grad/Reshape_1*
_output_shapes
: 
З
.gradients/add_30_grad/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*%
valueB"А         А   
q
.gradients/add_30_grad/BroadcastGradientArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 
с
+gradients/add_30_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/add_30_grad/BroadcastGradientArgs/s0.gradients/add_30_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:         :         
Д
+gradients/add_30_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
ж
gradients/add_30_grad/SumSumgradients/Log_26_grad/mul+gradients/add_30_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/add_30_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/add_30_grad/ReshapeReshapegradients/add_30_grad/Sum#gradients/add_30_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
j
&gradients/add_30_grad/tuple/group_depsNoOp^gradients/Log_26_grad/mul^gradients/add_30_grad/Reshape
▀
.gradients/add_30_grad/tuple/control_dependencyIdentitygradients/Log_26_grad/mul'^gradients/add_30_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_26_grad/mul*(
_output_shapes
:АА
╫
0gradients/add_30_grad/tuple/control_dependency_1Identitygradients/add_30_grad/Reshape'^gradients/add_30_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_30_grad/Reshape*
_output_shapes
: 
Д
+gradients/add_28_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
ж
gradients/add_28_grad/SumSumgradients/Log_24_grad/mul+gradients/add_28_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/add_28_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/add_28_grad/ReshapeReshapegradients/add_28_grad/Sum#gradients/add_28_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
j
&gradients/add_28_grad/tuple/group_depsNoOp^gradients/Log_24_grad/mul^gradients/add_28_grad/Reshape
▀
.gradients/add_28_grad/tuple/control_dependencyIdentitygradients/Log_24_grad/mul'^gradients/add_28_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_24_grad/mul*(
_output_shapes
:АА
╫
0gradients/add_28_grad/tuple/control_dependency_1Identitygradients/add_28_grad/Reshape'^gradients/add_28_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_28_grad/Reshape*
_output_shapes
: 
З
.gradients/add_37_grad/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*%
valueB"А         
   
q
.gradients/add_37_grad/BroadcastGradientArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 
с
+gradients/add_37_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/add_37_grad/BroadcastGradientArgs/s0.gradients/add_37_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:         :         
Д
+gradients/add_37_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
ж
gradients/add_37_grad/SumSumgradients/Log_32_grad/mul+gradients/add_37_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/add_37_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/add_37_grad/ReshapeReshapegradients/add_37_grad/Sum#gradients/add_37_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
j
&gradients/add_37_grad/tuple/group_depsNoOp^gradients/Log_32_grad/mul^gradients/add_37_grad/Reshape
▐
.gradients/add_37_grad/tuple/control_dependencyIdentitygradients/Log_32_grad/mul'^gradients/add_37_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_32_grad/mul*'
_output_shapes
:А

╫
0gradients/add_37_grad/tuple/control_dependency_1Identitygradients/add_37_grad/Reshape'^gradients/add_37_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_37_grad/Reshape*
_output_shapes
: 
Д
+gradients/add_35_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
ж
gradients/add_35_grad/SumSumgradients/Log_30_grad/mul+gradients/add_35_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
f
#gradients/add_35_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ч
gradients/add_35_grad/ReshapeReshapegradients/add_35_grad/Sum#gradients/add_35_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
j
&gradients/add_35_grad/tuple/group_depsNoOp^gradients/Log_30_grad/mul^gradients/add_35_grad/Reshape
▐
.gradients/add_35_grad/tuple/control_dependencyIdentitygradients/Log_30_grad/mul'^gradients/add_35_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Log_30_grad/mul*'
_output_shapes
:А

╫
0gradients/add_35_grad/tuple/control_dependency_1Identitygradients/add_35_grad/Reshape'^gradients/add_35_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_35_grad/Reshape*
_output_shapes
: 
~
%gradients/activation_3/mul_grad/ShapeShapeactivation_3/ReadVariableOp*
T0*
_output_shapes
: *
out_type0
{
'gradients/activation_3/mul_grad/Shape_1Shapeactivation_3/truediv*
T0*
_output_shapes
:*
out_type0
█
5gradients/activation_3/mul_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/activation_3/mul_grad/Shape'gradients/activation_3/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╣
#gradients/activation_3/mul_grad/MulMul<gradients/activation_3/add_1_grad/tuple/control_dependency_1activation_3/truediv*
T0*0
_output_shapes
:         А
╞
#gradients/activation_3/mul_grad/SumSum#gradients/activation_3/mul_grad/Mul5gradients/activation_3/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
н
'gradients/activation_3/mul_grad/ReshapeReshape#gradients/activation_3/mul_grad/Sum%gradients/activation_3/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
┬
%gradients/activation_3/mul_grad/Mul_1Mulactivation_3/ReadVariableOp<gradients/activation_3/add_1_grad/tuple/control_dependency_1*
T0*0
_output_shapes
:         А
╠
%gradients/activation_3/mul_grad/Sum_1Sum%gradients/activation_3/mul_grad/Mul_17gradients/activation_3/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
═
)gradients/activation_3/mul_grad/Reshape_1Reshape%gradients/activation_3/mul_grad/Sum_1'gradients/activation_3/mul_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:         А
О
0gradients/activation_3/mul_grad/tuple/group_depsNoOp(^gradients/activation_3/mul_grad/Reshape*^gradients/activation_3/mul_grad/Reshape_1
¤
8gradients/activation_3/mul_grad/tuple/control_dependencyIdentity'gradients/activation_3/mul_grad/Reshape1^gradients/activation_3/mul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/activation_3/mul_grad/Reshape*
_output_shapes
: 
Э
:gradients/activation_3/mul_grad/tuple/control_dependency_1Identity)gradients/activation_3/mul_grad/Reshape_11^gradients/activation_3/mul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/activation_3/mul_grad/Reshape_1*0
_output_shapes
:         А
`
gradients/Abs_1_grad/SignSignconv2d/mul_2*
T0*&
_output_shapes
:
Ъ
gradients/Abs_1_grad/mulMul-gradients/add_2_grad/tuple/control_dependencygradients/Abs_1_grad/Sign*
T0*&
_output_shapes
:
^
gradients/Abs_grad/SignSignconv2d/mul_2*
T0*&
_output_shapes
:
Ф
gradients/Abs_grad/mulMul+gradients/add_grad/tuple/control_dependencygradients/Abs_grad/Sign*
T0*&
_output_shapes
:
b
gradients/Abs_4_grad/SignSignconv2d_1/mul_2*
T0*&
_output_shapes
:

Ъ
gradients/Abs_4_grad/mulMul-gradients/add_9_grad/tuple/control_dependencygradients/Abs_4_grad/Sign*
T0*&
_output_shapes
:

b
gradients/Abs_3_grad/SignSignconv2d_1/mul_2*
T0*&
_output_shapes
:

Ъ
gradients/Abs_3_grad/mulMul-gradients/add_7_grad/tuple/control_dependencygradients/Abs_3_grad/Sign*
T0*&
_output_shapes
:

c
gradients/Abs_7_grad/SignSignconv2d_2/mul_2*
T0*'
_output_shapes
:1
А
Ь
gradients/Abs_7_grad/mulMul.gradients/add_16_grad/tuple/control_dependencygradients/Abs_7_grad/Sign*
T0*'
_output_shapes
:1
А
c
gradients/Abs_6_grad/SignSignconv2d_2/mul_2*
T0*'
_output_shapes
:1
А
Ь
gradients/Abs_6_grad/mulMul.gradients/add_14_grad/tuple/control_dependencygradients/Abs_6_grad/Sign*
T0*'
_output_shapes
:1
А
n
gradients/Abs_10_grad/SignSignconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
з
gradients/Abs_10_grad/mulMul.gradients/add_23_grad/tuple/control_dependencygradients/Abs_10_grad/Sign*
T0*0
_output_shapes
:         А
m
gradients/Abs_9_grad/SignSignconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
е
gradients/Abs_9_grad/mulMul.gradients/add_21_grad/tuple/control_dependencygradients/Abs_9_grad/Sign*
T0*0
_output_shapes
:         А
e
gradients/Abs_13_grad/SignSignconv2d_3/mul_2*
T0*(
_output_shapes
:АА
Я
gradients/Abs_13_grad/mulMul.gradients/add_30_grad/tuple/control_dependencygradients/Abs_13_grad/Sign*
T0*(
_output_shapes
:АА
e
gradients/Abs_12_grad/SignSignconv2d_3/mul_2*
T0*(
_output_shapes
:АА
Я
gradients/Abs_12_grad/mulMul.gradients/add_28_grad/tuple/control_dependencygradients/Abs_12_grad/Sign*
T0*(
_output_shapes
:АА
d
gradients/Abs_16_grad/SignSignconv2d_4/mul_2*
T0*'
_output_shapes
:А

Ю
gradients/Abs_16_grad/mulMul.gradients/add_37_grad/tuple/control_dependencygradients/Abs_16_grad/Sign*
T0*'
_output_shapes
:А

d
gradients/Abs_15_grad/SignSignconv2d_4/mul_2*
T0*'
_output_shapes
:А

Ю
gradients/Abs_15_grad/mulMul.gradients/add_35_grad/tuple/control_dependencygradients/Abs_15_grad/Sign*
T0*'
_output_shapes
:А

Р
gradients/AddN_1AddN.gradients/add_24_grad/tuple/control_dependency.gradients/add_22_grad/tuple/control_dependency:gradients/activation_3/add_1_grad/tuple/control_dependency*
N*
T0*0
_class&
$"loc:@gradients/add_24_grad/Reshape*
_output_shapes
: 
y
)gradients/activation_3/truediv_grad/ShapeShapeactivation_3/Log*
T0*
_output_shapes
:*
out_type0
n
+gradients/activation_3/truediv_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
ч
9gradients/activation_3/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/activation_3/truediv_grad/Shape+gradients/activation_3/truediv_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┴
+gradients/activation_3/truediv_grad/RealDivRealDiv:gradients/activation_3/mul_grad/tuple/control_dependency_1activation_3/Log_1*
T0*0
_output_shapes
:         А
╓
'gradients/activation_3/truediv_grad/SumSum+gradients/activation_3/truediv_grad/RealDiv9gradients/activation_3/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
╙
+gradients/activation_3/truediv_grad/ReshapeReshape'gradients/activation_3/truediv_grad/Sum)gradients/activation_3/truediv_grad/Shape*
T0*
Tshape0*0
_output_shapes
:         А
{
'gradients/activation_3/truediv_grad/NegNegactivation_3/Log*
T0*0
_output_shapes
:         А
░
-gradients/activation_3/truediv_grad/RealDiv_1RealDiv'gradients/activation_3/truediv_grad/Negactivation_3/Log_1*
T0*0
_output_shapes
:         А
╢
-gradients/activation_3/truediv_grad/RealDiv_2RealDiv-gradients/activation_3/truediv_grad/RealDiv_1activation_3/Log_1*
T0*0
_output_shapes
:         А
╘
'gradients/activation_3/truediv_grad/mulMul:gradients/activation_3/mul_grad/tuple/control_dependency_1-gradients/activation_3/truediv_grad/RealDiv_2*
T0*0
_output_shapes
:         А
╓
)gradients/activation_3/truediv_grad/Sum_1Sum'gradients/activation_3/truediv_grad/mul;gradients/activation_3/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
┐
-gradients/activation_3/truediv_grad/Reshape_1Reshape)gradients/activation_3/truediv_grad/Sum_1+gradients/activation_3/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ъ
4gradients/activation_3/truediv_grad/tuple/group_depsNoOp,^gradients/activation_3/truediv_grad/Reshape.^gradients/activation_3/truediv_grad/Reshape_1
з
<gradients/activation_3/truediv_grad/tuple/control_dependencyIdentity+gradients/activation_3/truediv_grad/Reshape5^gradients/activation_3/truediv_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/activation_3/truediv_grad/Reshape*0
_output_shapes
:         А
У
>gradients/activation_3/truediv_grad/tuple/control_dependency_1Identity-gradients/activation_3/truediv_grad/Reshape_15^gradients/activation_3/truediv_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/activation_3/truediv_grad/Reshape_1*
_output_shapes
: 
·
gradients/AddN_2AddN)gradients/activation_3/Relu_grad/ReluGradgradients/Abs_10_grad/mulgradients/Abs_9_grad/mul*
N*
T0*<
_class2
0.loc:@gradients/activation_3/Relu_grad/ReluGrad*0
_output_shapes
:         А
д
%gradients/conv2d_2/Conv2D_grad/ShapeNShapeNreshape/Reshapeconv2d_2/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::*
out_type0
ф
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_2/Conv2D/ReadVariableOpgradients/AddN_2*
T0*/
_output_shapes
:         1
*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
╤
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterreshape/Reshape'gradients/conv2d_2/Conv2D_grad/ShapeN:1gradients/AddN_2*
T0*'
_output_shapes
:1
А*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
в
/gradients/conv2d_2/Conv2D_grad/tuple/group_depsNoOp4^gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter3^gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput
к
7gradients/conv2d_2/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_2/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         1

ж
9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_2/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:1
А
С
gradients/AddN_3AddN;gradients/conv2d_3/Conv2D_1_grad/tuple/control_dependency_1gradients/Abs_13_grad/mulgradients/Abs_12_grad/mul*
N*
T0*H
_class>
<:loc:@gradients/conv2d_3/Conv2D_1_grad/Conv2DBackpropFilter*(
_output_shapes
:АА
{
!gradients/conv2d_3/mul_2_grad/MulMulgradients/AddN_3conv2d_3/pow*
T0*(
_output_shapes
:АА
Д
#gradients/conv2d_3/mul_2_grad/Mul_1Mulgradients/AddN_3conv2d_3/lp_weights*
T0*(
_output_shapes
:АА
А
.gradients/conv2d_3/mul_2_grad/tuple/group_depsNoOp"^gradients/conv2d_3/mul_2_grad/Mul$^gradients/conv2d_3/mul_2_grad/Mul_1
 
6gradients/conv2d_3/mul_2_grad/tuple/control_dependencyIdentity!gradients/conv2d_3/mul_2_grad/Mul/^gradients/conv2d_3/mul_2_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_3/mul_2_grad/Mul*(
_output_shapes
:АА
Е
8gradients/conv2d_3/mul_2_grad/tuple/control_dependency_1Identity#gradients/conv2d_3/mul_2_grad/Mul_1/^gradients/conv2d_3/mul_2_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_3/mul_2_grad/Mul_1*(
_output_shapes
:АА
Р
gradients/AddN_4AddN;gradients/conv2d_4/Conv2D_1_grad/tuple/control_dependency_1gradients/Abs_16_grad/mulgradients/Abs_15_grad/mul*
N*
T0*H
_class>
<:loc:@gradients/conv2d_4/Conv2D_1_grad/Conv2DBackpropFilter*'
_output_shapes
:А

z
!gradients/conv2d_4/mul_2_grad/MulMulgradients/AddN_4conv2d_4/pow*
T0*'
_output_shapes
:А

Г
#gradients/conv2d_4/mul_2_grad/Mul_1Mulgradients/AddN_4conv2d_4/lp_weights*
T0*'
_output_shapes
:А

А
.gradients/conv2d_4/mul_2_grad/tuple/group_depsNoOp"^gradients/conv2d_4/mul_2_grad/Mul$^gradients/conv2d_4/mul_2_grad/Mul_1
■
6gradients/conv2d_4/mul_2_grad/tuple/control_dependencyIdentity!gradients/conv2d_4/mul_2_grad/Mul/^gradients/conv2d_4/mul_2_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_4/mul_2_grad/Mul*'
_output_shapes
:А

Д
8gradients/conv2d_4/mul_2_grad/tuple/control_dependency_1Identity#gradients/conv2d_4/mul_2_grad/Mul_1/^gradients/conv2d_4/mul_2_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_4/mul_2_grad/Mul_1*'
_output_shapes
:А

Л
gradients/AddN_5AddN-gradients/mul_7_grad/tuple/control_dependency-gradients/mul_6_grad/tuple/control_dependency8gradients/activation_3/mul_grad/tuple/control_dependency*
N*
T0*/
_class%
#!loc:@gradients/mul_7_grad/Reshape*
_output_shapes
: 
─
*gradients/activation_3/Log_grad/Reciprocal
Reciprocalactivation_3/add=^gradients/activation_3/truediv_grad/tuple/control_dependency*
T0*0
_output_shapes
:         А
╧
#gradients/activation_3/Log_grad/mulMul<gradients/activation_3/truediv_grad/tuple/control_dependency*gradients/activation_3/Log_grad/Reciprocal*
T0*0
_output_shapes
:         А
s
$gradients/reshape/Reshape_grad/ShapeShapeflatten/Reshape*
T0*
_output_shapes
:*
out_type0
╤
&gradients/reshape/Reshape_grad/ReshapeReshape7gradients/conv2d_2/Conv2D_grad/tuple/control_dependency$gradients/reshape/Reshape_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ъ
▓
gradients/conv2d_3/pow_grad/mulMul8gradients/conv2d_3/mul_2_grad/tuple/control_dependency_1conv2d_3/differentiable_round*
T0*(
_output_shapes
:АА
f
!gradients/conv2d_3/pow_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ы
gradients/conv2d_3/pow_grad/subSubconv2d_3/differentiable_round!gradients/conv2d_3/pow_grad/sub/y*
T0*(
_output_shapes
:АА
К
gradients/conv2d_3/pow_grad/PowPowconv2d_3/pow/xgradients/conv2d_3/pow_grad/sub*
T0*(
_output_shapes
:АА
Э
!gradients/conv2d_3/pow_grad/mul_1Mulgradients/conv2d_3/pow_grad/mulgradients/conv2d_3/pow_grad/Pow*
T0*(
_output_shapes
:АА
К
1gradients/conv2d_3/pow_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
║
gradients/conv2d_3/pow_grad/SumSum!gradients/conv2d_3/pow_grad/mul_11gradients/conv2d_3/pow_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
l
)gradients/conv2d_3/pow_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
й
#gradients/conv2d_3/pow_grad/ReshapeReshapegradients/conv2d_3/pow_grad/Sum)gradients/conv2d_3/pow_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
j
%gradients/conv2d_3/pow_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
Ж
#gradients/conv2d_3/pow_grad/GreaterGreaterconv2d_3/pow/x%gradients/conv2d_3/pow_grad/Greater/y*
T0*
_output_shapes
: 
n
+gradients/conv2d_3/pow_grad/ones_like/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
p
+gradients/conv2d_3/pow_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
║
%gradients/conv2d_3/pow_grad/ones_likeFill+gradients/conv2d_3/pow_grad/ones_like/Shape+gradients/conv2d_3/pow_grad/ones_like/Const*
T0*
_output_shapes
: *

index_type0
й
"gradients/conv2d_3/pow_grad/SelectSelect#gradients/conv2d_3/pow_grad/Greaterconv2d_3/pow/x%gradients/conv2d_3/pow_grad/ones_like*
T0*
_output_shapes
: 
k
gradients/conv2d_3/pow_grad/LogLog"gradients/conv2d_3/pow_grad/Select*
T0*
_output_shapes
: 
k
&gradients/conv2d_3/pow_grad/zeros_likeConst*
_output_shapes
: *
dtype0*
valueB
 *    
╜
$gradients/conv2d_3/pow_grad/Select_1Select#gradients/conv2d_3/pow_grad/Greatergradients/conv2d_3/pow_grad/Log&gradients/conv2d_3/pow_grad/zeros_like*
T0*
_output_shapes
: 
г
!gradients/conv2d_3/pow_grad/mul_2Mul8gradients/conv2d_3/mul_2_grad/tuple/control_dependency_1conv2d_3/pow*
T0*(
_output_shapes
:АА
д
!gradients/conv2d_3/pow_grad/mul_3Mul!gradients/conv2d_3/pow_grad/mul_2$gradients/conv2d_3/pow_grad/Select_1*
T0*(
_output_shapes
:АА
~
,gradients/conv2d_3/pow_grad/tuple/group_depsNoOp$^gradients/conv2d_3/pow_grad/Reshape"^gradients/conv2d_3/pow_grad/mul_3
э
4gradients/conv2d_3/pow_grad/tuple/control_dependencyIdentity#gradients/conv2d_3/pow_grad/Reshape-^gradients/conv2d_3/pow_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_3/pow_grad/Reshape*
_output_shapes
: 
¤
6gradients/conv2d_3/pow_grad/tuple/control_dependency_1Identity!gradients/conv2d_3/pow_grad/mul_3-^gradients/conv2d_3/pow_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_3/pow_grad/mul_3*(
_output_shapes
:АА
▒
gradients/conv2d_4/pow_grad/mulMul8gradients/conv2d_4/mul_2_grad/tuple/control_dependency_1conv2d_4/differentiable_round*
T0*'
_output_shapes
:А

f
!gradients/conv2d_4/pow_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ъ
gradients/conv2d_4/pow_grad/subSubconv2d_4/differentiable_round!gradients/conv2d_4/pow_grad/sub/y*
T0*'
_output_shapes
:А

Й
gradients/conv2d_4/pow_grad/PowPowconv2d_4/pow/xgradients/conv2d_4/pow_grad/sub*
T0*'
_output_shapes
:А

Ь
!gradients/conv2d_4/pow_grad/mul_1Mulgradients/conv2d_4/pow_grad/mulgradients/conv2d_4/pow_grad/Pow*
T0*'
_output_shapes
:А

К
1gradients/conv2d_4/pow_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
║
gradients/conv2d_4/pow_grad/SumSum!gradients/conv2d_4/pow_grad/mul_11gradients/conv2d_4/pow_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
l
)gradients/conv2d_4/pow_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
й
#gradients/conv2d_4/pow_grad/ReshapeReshapegradients/conv2d_4/pow_grad/Sum)gradients/conv2d_4/pow_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
j
%gradients/conv2d_4/pow_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
Ж
#gradients/conv2d_4/pow_grad/GreaterGreaterconv2d_4/pow/x%gradients/conv2d_4/pow_grad/Greater/y*
T0*
_output_shapes
: 
n
+gradients/conv2d_4/pow_grad/ones_like/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
p
+gradients/conv2d_4/pow_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
║
%gradients/conv2d_4/pow_grad/ones_likeFill+gradients/conv2d_4/pow_grad/ones_like/Shape+gradients/conv2d_4/pow_grad/ones_like/Const*
T0*
_output_shapes
: *

index_type0
й
"gradients/conv2d_4/pow_grad/SelectSelect#gradients/conv2d_4/pow_grad/Greaterconv2d_4/pow/x%gradients/conv2d_4/pow_grad/ones_like*
T0*
_output_shapes
: 
k
gradients/conv2d_4/pow_grad/LogLog"gradients/conv2d_4/pow_grad/Select*
T0*
_output_shapes
: 
k
&gradients/conv2d_4/pow_grad/zeros_likeConst*
_output_shapes
: *
dtype0*
valueB
 *    
╜
$gradients/conv2d_4/pow_grad/Select_1Select#gradients/conv2d_4/pow_grad/Greatergradients/conv2d_4/pow_grad/Log&gradients/conv2d_4/pow_grad/zeros_like*
T0*
_output_shapes
: 
в
!gradients/conv2d_4/pow_grad/mul_2Mul8gradients/conv2d_4/mul_2_grad/tuple/control_dependency_1conv2d_4/pow*
T0*'
_output_shapes
:А

г
!gradients/conv2d_4/pow_grad/mul_3Mul!gradients/conv2d_4/pow_grad/mul_2$gradients/conv2d_4/pow_grad/Select_1*
T0*'
_output_shapes
:А

~
,gradients/conv2d_4/pow_grad/tuple/group_depsNoOp$^gradients/conv2d_4/pow_grad/Reshape"^gradients/conv2d_4/pow_grad/mul_3
э
4gradients/conv2d_4/pow_grad/tuple/control_dependencyIdentity#gradients/conv2d_4/pow_grad/Reshape-^gradients/conv2d_4/pow_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_4/pow_grad/Reshape*
_output_shapes
: 
№
6gradients/conv2d_4/pow_grad/tuple/control_dependency_1Identity!gradients/conv2d_4/pow_grad/mul_3-^gradients/conv2d_4/pow_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_4/pow_grad/mul_3*'
_output_shapes
:А

u
%gradients/activation_3/add_grad/ShapeShapeactivation_3/Abs*
T0*
_output_shapes
:*
out_type0
w
'gradients/activation_3/add_grad/Shape_1Shapeactivation_3/add/y*
T0*
_output_shapes
: *
out_type0
█
5gradients/activation_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/activation_3/add_grad/Shape'gradients/activation_3/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╞
#gradients/activation_3/add_grad/SumSum#gradients/activation_3/Log_grad/mul5gradients/activation_3/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
╟
'gradients/activation_3/add_grad/ReshapeReshape#gradients/activation_3/add_grad/Sum%gradients/activation_3/add_grad/Shape*
T0*
Tshape0*0
_output_shapes
:         А
╩
%gradients/activation_3/add_grad/Sum_1Sum#gradients/activation_3/Log_grad/mul7gradients/activation_3/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
│
)gradients/activation_3/add_grad/Reshape_1Reshape%gradients/activation_3/add_grad/Sum_1'gradients/activation_3/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
О
0gradients/activation_3/add_grad/tuple/group_depsNoOp(^gradients/activation_3/add_grad/Reshape*^gradients/activation_3/add_grad/Reshape_1
Ч
8gradients/activation_3/add_grad/tuple/control_dependencyIdentity'gradients/activation_3/add_grad/Reshape1^gradients/activation_3/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/activation_3/add_grad/Reshape*0
_output_shapes
:         А
Г
:gradients/activation_3/add_grad/tuple/control_dependency_1Identity)gradients/activation_3/add_grad/Reshape_11^gradients/activation_3/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/activation_3/add_grad/Reshape_1*
_output_shapes
: 
{
$gradients/flatten/Reshape_grad/ShapeShapemax_pooling2d_1/MaxPool*
T0*
_output_shapes
:*
out_type0
╟
&gradients/flatten/Reshape_grad/ReshapeReshape&gradients/reshape/Reshape_grad/Reshape$gradients/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         

|
$gradients/activation_3/Abs_grad/SignSignactivation_3/Relu_1*
T0*0
_output_shapes
:         А
┼
#gradients/activation_3/Abs_grad/mulMul8gradients/activation_3/add_grad/tuple/control_dependency$gradients/activation_3/Abs_grad/Sign*
T0*0
_output_shapes
:         А
б
2gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_2/Relumax_pooling2d_1/MaxPool&gradients/flatten/Reshape_grad/Reshape*
T0*/
_output_shapes
:         
*
data_formatNHWC*
ksize
*
paddingSAME*
strides

М
3gradients/conv2d_3/add_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
╙
!gradients/conv2d_3/add_1_grad/SumSum6gradients/conv2d_3/pow_grad/tuple/control_dependency_13gradients/conv2d_3/add_1_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
n
+gradients/conv2d_3/add_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
п
%gradients/conv2d_3/add_1_grad/ReshapeReshape!gradients/conv2d_3/add_1_grad/Sum+gradients/conv2d_3/add_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
Ч
.gradients/conv2d_3/add_1_grad/tuple/group_depsNoOp&^gradients/conv2d_3/add_1_grad/Reshape7^gradients/conv2d_3/pow_grad/tuple/control_dependency_1
ї
6gradients/conv2d_3/add_1_grad/tuple/control_dependencyIdentity%gradients/conv2d_3/add_1_grad/Reshape/^gradients/conv2d_3/add_1_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/conv2d_3/add_1_grad/Reshape*
_output_shapes
: 
Ц
8gradients/conv2d_3/add_1_grad/tuple/control_dependency_1Identity6gradients/conv2d_3/pow_grad/tuple/control_dependency_1/^gradients/conv2d_3/add_1_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_3/pow_grad/mul_3*(
_output_shapes
:АА
М
3gradients/conv2d_4/add_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
╙
!gradients/conv2d_4/add_1_grad/SumSum6gradients/conv2d_4/pow_grad/tuple/control_dependency_13gradients/conv2d_4/add_1_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
n
+gradients/conv2d_4/add_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
п
%gradients/conv2d_4/add_1_grad/ReshapeReshape!gradients/conv2d_4/add_1_grad/Sum+gradients/conv2d_4/add_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
Ч
.gradients/conv2d_4/add_1_grad/tuple/group_depsNoOp&^gradients/conv2d_4/add_1_grad/Reshape7^gradients/conv2d_4/pow_grad/tuple/control_dependency_1
ї
6gradients/conv2d_4/add_1_grad/tuple/control_dependencyIdentity%gradients/conv2d_4/add_1_grad/Reshape/^gradients/conv2d_4/add_1_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/conv2d_4/add_1_grad/Reshape*
_output_shapes
: 
Х
8gradients/conv2d_4/add_1_grad/tuple/control_dependency_1Identity6gradients/conv2d_4/pow_grad/tuple/control_dependency_1/^gradients/conv2d_4/add_1_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_4/pow_grad/mul_3*'
_output_shapes
:А

м
+gradients/activation_3/Relu_1_grad/ReluGradReluGrad#gradients/activation_3/Abs_grad/mulactivation_3/Relu_1*
T0*0
_output_shapes
:         А
╢
)gradients/activation_2/Relu_grad/ReluGradReluGrad2gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradactivation_2/Relu*
T0*/
_output_shapes
:         

е
gradients/conv2d_3/mul_grad/MulMul8gradients/conv2d_3/add_1_grad/tuple/control_dependency_1conv2d_3/truediv*
T0*(
_output_shapes
:АА
К
1gradients/conv2d_3/mul_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
╕
gradients/conv2d_3/mul_grad/SumSumgradients/conv2d_3/mul_grad/Mul1gradients/conv2d_3/mul_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
l
)gradients/conv2d_3/mul_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
й
#gradients/conv2d_3/mul_grad/ReshapeReshapegradients/conv2d_3/mul_grad/Sum)gradients/conv2d_3/mul_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
о
!gradients/conv2d_3/mul_grad/Mul_1Mulconv2d_3/ReadVariableOp8gradients/conv2d_3/add_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:АА
~
,gradients/conv2d_3/mul_grad/tuple/group_depsNoOp"^gradients/conv2d_3/mul_grad/Mul_1$^gradients/conv2d_3/mul_grad/Reshape
э
4gradients/conv2d_3/mul_grad/tuple/control_dependencyIdentity#gradients/conv2d_3/mul_grad/Reshape-^gradients/conv2d_3/mul_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_3/mul_grad/Reshape*
_output_shapes
: 
¤
6gradients/conv2d_3/mul_grad/tuple/control_dependency_1Identity!gradients/conv2d_3/mul_grad/Mul_1-^gradients/conv2d_3/mul_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_3/mul_grad/Mul_1*(
_output_shapes
:АА
д
gradients/conv2d_4/mul_grad/MulMul8gradients/conv2d_4/add_1_grad/tuple/control_dependency_1conv2d_4/truediv*
T0*'
_output_shapes
:А

К
1gradients/conv2d_4/mul_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
╕
gradients/conv2d_4/mul_grad/SumSumgradients/conv2d_4/mul_grad/Mul1gradients/conv2d_4/mul_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
l
)gradients/conv2d_4/mul_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
й
#gradients/conv2d_4/mul_grad/ReshapeReshapegradients/conv2d_4/mul_grad/Sum)gradients/conv2d_4/mul_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
н
!gradients/conv2d_4/mul_grad/Mul_1Mulconv2d_4/ReadVariableOp8gradients/conv2d_4/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:А

~
,gradients/conv2d_4/mul_grad/tuple/group_depsNoOp"^gradients/conv2d_4/mul_grad/Mul_1$^gradients/conv2d_4/mul_grad/Reshape
э
4gradients/conv2d_4/mul_grad/tuple/control_dependencyIdentity#gradients/conv2d_4/mul_grad/Reshape-^gradients/conv2d_4/mul_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_4/mul_grad/Reshape*
_output_shapes
: 
№
6gradients/conv2d_4/mul_grad/tuple/control_dependency_1Identity!gradients/conv2d_4/mul_grad/Mul_1-^gradients/conv2d_4/mul_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_4/mul_grad/Mul_1*'
_output_shapes
:А

б
'gradients/conv2d_2/Conv2D_1_grad/ShapeNShapeNreshape/reshape_requantizeconv2d_2/mul_2*
N*
T0* 
_output_shapes
::*
out_type0
є
4gradients/conv2d_2/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/conv2d_2/Conv2D_1_grad/ShapeNconv2d_2/mul_2+gradients/activation_3/Relu_1_grad/ReluGrad*
T0*/
_output_shapes
:         1
*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
√
5gradients/conv2d_2/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterreshape/reshape_requantize)gradients/conv2d_2/Conv2D_1_grad/ShapeN:1+gradients/activation_3/Relu_1_grad/ReluGrad*
T0*'
_output_shapes
:1
А*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
и
1gradients/conv2d_2/Conv2D_1_grad/tuple/group_depsNoOp6^gradients/conv2d_2/Conv2D_1_grad/Conv2DBackpropFilter5^gradients/conv2d_2/Conv2D_1_grad/Conv2DBackpropInput
▓
9gradients/conv2d_2/Conv2D_1_grad/tuple/control_dependencyIdentity4gradients/conv2d_2/Conv2D_1_grad/Conv2DBackpropInput2^gradients/conv2d_2/Conv2D_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/conv2d_2/Conv2D_1_grad/Conv2DBackpropInput*/
_output_shapes
:         1

о
;gradients/conv2d_2/Conv2D_1_grad/tuple/control_dependency_1Identity5gradients/conv2d_2/Conv2D_1_grad/Conv2DBackpropFilter2^gradients/conv2d_2/Conv2D_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/conv2d_2/Conv2D_1_grad/Conv2DBackpropFilter*'
_output_shapes
:1
А
к
%gradients/conv2d_1/Conv2D_grad/ShapeNShapeNmax_pooling2d/MaxPoolconv2d_1/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::*
out_type0
№
2gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_1/Conv2D_grad/ShapeNconv2d_1/Conv2D/ReadVariableOp)gradients/activation_2/Relu_grad/ReluGrad*
T0*/
_output_shapes
:         *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
ю
3gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool'gradients/conv2d_1/Conv2D_grad/ShapeN:1)gradients/activation_2/Relu_grad/ReluGrad*
T0*&
_output_shapes
:
*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
в
/gradients/conv2d_1/Conv2D_grad/tuple/group_depsNoOp4^gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter3^gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput
к
7gradients/conv2d_1/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         
е
9gradients/conv2d_1/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:

М
gradients/AddN_6AddN.gradients/add_31_grad/tuple/control_dependency.gradients/add_29_grad/tuple/control_dependency6gradients/conv2d_3/add_1_grad/tuple/control_dependency*
N*
T0*0
_class&
$"loc:@gradients/add_31_grad/Reshape*
_output_shapes
: 
~
%gradients/conv2d_3/truediv_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
j
'gradients/conv2d_3/truediv_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
█
5gradients/conv2d_3/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/conv2d_3/truediv_grad/Shape'gradients/conv2d_3/truediv_grad/Shape_1*
T0*2
_output_shapes 
:         :         
н
'gradients/conv2d_3/truediv_grad/RealDivRealDiv6gradients/conv2d_3/mul_grad/tuple/control_dependency_1conv2d_3/Log_1*
T0*(
_output_shapes
:АА
╥
#gradients/conv2d_3/truediv_grad/SumSum'gradients/conv2d_3/truediv_grad/RealDiv5gradients/conv2d_3/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0* 
_output_shapes
:
АА*
	keep_dims( 
┐
'gradients/conv2d_3/truediv_grad/ReshapeReshape#gradients/conv2d_3/truediv_grad/Sum%gradients/conv2d_3/truediv_grad/Shape*
T0*
Tshape0*(
_output_shapes
:АА
k
#gradients/conv2d_3/truediv_grad/NegNegconv2d_3/Log*
T0*(
_output_shapes
:АА
Ь
)gradients/conv2d_3/truediv_grad/RealDiv_1RealDiv#gradients/conv2d_3/truediv_grad/Negconv2d_3/Log_1*
T0*(
_output_shapes
:АА
в
)gradients/conv2d_3/truediv_grad/RealDiv_2RealDiv)gradients/conv2d_3/truediv_grad/RealDiv_1conv2d_3/Log_1*
T0*(
_output_shapes
:АА
└
#gradients/conv2d_3/truediv_grad/mulMul6gradients/conv2d_3/mul_grad/tuple/control_dependency_1)gradients/conv2d_3/truediv_grad/RealDiv_2*
T0*(
_output_shapes
:АА
╚
%gradients/conv2d_3/truediv_grad/Sum_1Sum#gradients/conv2d_3/truediv_grad/mul7gradients/conv2d_3/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
│
)gradients/conv2d_3/truediv_grad/Reshape_1Reshape%gradients/conv2d_3/truediv_grad/Sum_1'gradients/conv2d_3/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
О
0gradients/conv2d_3/truediv_grad/tuple/group_depsNoOp(^gradients/conv2d_3/truediv_grad/Reshape*^gradients/conv2d_3/truediv_grad/Reshape_1
П
8gradients/conv2d_3/truediv_grad/tuple/control_dependencyIdentity'gradients/conv2d_3/truediv_grad/Reshape1^gradients/conv2d_3/truediv_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/conv2d_3/truediv_grad/Reshape*(
_output_shapes
:АА
Г
:gradients/conv2d_3/truediv_grad/tuple/control_dependency_1Identity)gradients/conv2d_3/truediv_grad/Reshape_11^gradients/conv2d_3/truediv_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/conv2d_3/truediv_grad/Reshape_1*
_output_shapes
: 
М
gradients/AddN_7AddN.gradients/add_38_grad/tuple/control_dependency.gradients/add_36_grad/tuple/control_dependency6gradients/conv2d_4/add_1_grad/tuple/control_dependency*
N*
T0*0
_class&
$"loc:@gradients/add_38_grad/Reshape*
_output_shapes
: 
~
%gradients/conv2d_4/truediv_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
j
'gradients/conv2d_4/truediv_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
█
5gradients/conv2d_4/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/conv2d_4/truediv_grad/Shape'gradients/conv2d_4/truediv_grad/Shape_1*
T0*2
_output_shapes 
:         :         
м
'gradients/conv2d_4/truediv_grad/RealDivRealDiv6gradients/conv2d_4/mul_grad/tuple/control_dependency_1conv2d_4/Log_1*
T0*'
_output_shapes
:А

╤
#gradients/conv2d_4/truediv_grad/SumSum'gradients/conv2d_4/truediv_grad/RealDiv5gradients/conv2d_4/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:	А
*
	keep_dims( 
╛
'gradients/conv2d_4/truediv_grad/ReshapeReshape#gradients/conv2d_4/truediv_grad/Sum%gradients/conv2d_4/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:А

j
#gradients/conv2d_4/truediv_grad/NegNegconv2d_4/Log*
T0*'
_output_shapes
:А

Ы
)gradients/conv2d_4/truediv_grad/RealDiv_1RealDiv#gradients/conv2d_4/truediv_grad/Negconv2d_4/Log_1*
T0*'
_output_shapes
:А

б
)gradients/conv2d_4/truediv_grad/RealDiv_2RealDiv)gradients/conv2d_4/truediv_grad/RealDiv_1conv2d_4/Log_1*
T0*'
_output_shapes
:А

┐
#gradients/conv2d_4/truediv_grad/mulMul6gradients/conv2d_4/mul_grad/tuple/control_dependency_1)gradients/conv2d_4/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:А

╚
%gradients/conv2d_4/truediv_grad/Sum_1Sum#gradients/conv2d_4/truediv_grad/mul7gradients/conv2d_4/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
│
)gradients/conv2d_4/truediv_grad/Reshape_1Reshape%gradients/conv2d_4/truediv_grad/Sum_1'gradients/conv2d_4/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
О
0gradients/conv2d_4/truediv_grad/tuple/group_depsNoOp(^gradients/conv2d_4/truediv_grad/Reshape*^gradients/conv2d_4/truediv_grad/Reshape_1
О
8gradients/conv2d_4/truediv_grad/tuple/control_dependencyIdentity'gradients/conv2d_4/truediv_grad/Reshape1^gradients/conv2d_4/truediv_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/conv2d_4/truediv_grad/Reshape*'
_output_shapes
:А

Г
:gradients/conv2d_4/truediv_grad/tuple/control_dependency_1Identity)gradients/conv2d_4/truediv_grad/Reshape_11^gradients/conv2d_4/truediv_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/conv2d_4/truediv_grad/Reshape_1*
_output_shapes
: 
ё
gradients/AddN_8AddNgradients/Abs_7_grad/mulgradients/Abs_6_grad/mul;gradients/conv2d_2/Conv2D_1_grad/tuple/control_dependency_1*
N*
T0*+
_class!
loc:@gradients/Abs_7_grad/mul*'
_output_shapes
:1
А
z
!gradients/conv2d_2/mul_2_grad/MulMulgradients/AddN_8conv2d_2/pow*
T0*'
_output_shapes
:1
А
Г
#gradients/conv2d_2/mul_2_grad/Mul_1Mulgradients/AddN_8conv2d_2/lp_weights*
T0*'
_output_shapes
:1
А
А
.gradients/conv2d_2/mul_2_grad/tuple/group_depsNoOp"^gradients/conv2d_2/mul_2_grad/Mul$^gradients/conv2d_2/mul_2_grad/Mul_1
■
6gradients/conv2d_2/mul_2_grad/tuple/control_dependencyIdentity!gradients/conv2d_2/mul_2_grad/Mul/^gradients/conv2d_2/mul_2_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_2/mul_2_grad/Mul*'
_output_shapes
:1
А
Д
8gradients/conv2d_2/mul_2_grad/tuple/control_dependency_1Identity#gradients/conv2d_2/mul_2_grad/Mul_1/^gradients/conv2d_2/mul_2_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_2/mul_2_grad/Mul_1*'
_output_shapes
:1
А
о
0gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_1/Relumax_pooling2d/MaxPool7gradients/conv2d_1/Conv2D_grad/tuple/control_dependency*
T0*/
_output_shapes
:         *
data_formatNHWC*
ksize
*
paddingSAME*
strides

З
gradients/AddN_9AddN-gradients/mul_9_grad/tuple/control_dependency-gradients/mul_8_grad/tuple/control_dependency4gradients/conv2d_3/mul_grad/tuple/control_dependency*
N*
T0*/
_class%
#!loc:@gradients/mul_9_grad/Reshape*
_output_shapes
: 
░
&gradients/conv2d_3/Log_grad/Reciprocal
Reciprocalconv2d_3/add9^gradients/conv2d_3/truediv_grad/tuple/control_dependency*
T0*(
_output_shapes
:АА
╗
gradients/conv2d_3/Log_grad/mulMul8gradients/conv2d_3/truediv_grad/tuple/control_dependency&gradients/conv2d_3/Log_grad/Reciprocal*
T0*(
_output_shapes
:АА
Л
gradients/AddN_10AddN.gradients/mul_11_grad/tuple/control_dependency.gradients/mul_10_grad/tuple/control_dependency4gradients/conv2d_4/mul_grad/tuple/control_dependency*
N*
T0*0
_class&
$"loc:@gradients/mul_11_grad/Reshape*
_output_shapes
: 
п
&gradients/conv2d_4/Log_grad/Reciprocal
Reciprocalconv2d_4/add9^gradients/conv2d_4/truediv_grad/tuple/control_dependency*
T0*'
_output_shapes
:А

║
gradients/conv2d_4/Log_grad/mulMul8gradients/conv2d_4/truediv_grad/tuple/control_dependency&gradients/conv2d_4/Log_grad/Reciprocal*
T0*'
_output_shapes
:А

w
&gradients/reshape/Reshape_2_grad/ShapeShapeflatten/Reshape_1*
T0*
_output_shapes
:*
out_type0
╫
(gradients/reshape/Reshape_2_grad/ReshapeReshape9gradients/conv2d_2/Conv2D_1_grad/tuple/control_dependency&gradients/reshape/Reshape_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ъ
▒
gradients/conv2d_2/pow_grad/mulMul8gradients/conv2d_2/mul_2_grad/tuple/control_dependency_1conv2d_2/differentiable_round*
T0*'
_output_shapes
:1
А
f
!gradients/conv2d_2/pow_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ъ
gradients/conv2d_2/pow_grad/subSubconv2d_2/differentiable_round!gradients/conv2d_2/pow_grad/sub/y*
T0*'
_output_shapes
:1
А
Й
gradients/conv2d_2/pow_grad/PowPowconv2d_2/pow/xgradients/conv2d_2/pow_grad/sub*
T0*'
_output_shapes
:1
А
Ь
!gradients/conv2d_2/pow_grad/mul_1Mulgradients/conv2d_2/pow_grad/mulgradients/conv2d_2/pow_grad/Pow*
T0*'
_output_shapes
:1
А
К
1gradients/conv2d_2/pow_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
║
gradients/conv2d_2/pow_grad/SumSum!gradients/conv2d_2/pow_grad/mul_11gradients/conv2d_2/pow_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
l
)gradients/conv2d_2/pow_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
й
#gradients/conv2d_2/pow_grad/ReshapeReshapegradients/conv2d_2/pow_grad/Sum)gradients/conv2d_2/pow_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
j
%gradients/conv2d_2/pow_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
Ж
#gradients/conv2d_2/pow_grad/GreaterGreaterconv2d_2/pow/x%gradients/conv2d_2/pow_grad/Greater/y*
T0*
_output_shapes
: 
n
+gradients/conv2d_2/pow_grad/ones_like/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
p
+gradients/conv2d_2/pow_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
║
%gradients/conv2d_2/pow_grad/ones_likeFill+gradients/conv2d_2/pow_grad/ones_like/Shape+gradients/conv2d_2/pow_grad/ones_like/Const*
T0*
_output_shapes
: *

index_type0
й
"gradients/conv2d_2/pow_grad/SelectSelect#gradients/conv2d_2/pow_grad/Greaterconv2d_2/pow/x%gradients/conv2d_2/pow_grad/ones_like*
T0*
_output_shapes
: 
k
gradients/conv2d_2/pow_grad/LogLog"gradients/conv2d_2/pow_grad/Select*
T0*
_output_shapes
: 
k
&gradients/conv2d_2/pow_grad/zeros_likeConst*
_output_shapes
: *
dtype0*
valueB
 *    
╜
$gradients/conv2d_2/pow_grad/Select_1Select#gradients/conv2d_2/pow_grad/Greatergradients/conv2d_2/pow_grad/Log&gradients/conv2d_2/pow_grad/zeros_like*
T0*
_output_shapes
: 
в
!gradients/conv2d_2/pow_grad/mul_2Mul8gradients/conv2d_2/mul_2_grad/tuple/control_dependency_1conv2d_2/pow*
T0*'
_output_shapes
:1
А
г
!gradients/conv2d_2/pow_grad/mul_3Mul!gradients/conv2d_2/pow_grad/mul_2$gradients/conv2d_2/pow_grad/Select_1*
T0*'
_output_shapes
:1
А
~
,gradients/conv2d_2/pow_grad/tuple/group_depsNoOp$^gradients/conv2d_2/pow_grad/Reshape"^gradients/conv2d_2/pow_grad/mul_3
э
4gradients/conv2d_2/pow_grad/tuple/control_dependencyIdentity#gradients/conv2d_2/pow_grad/Reshape-^gradients/conv2d_2/pow_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_2/pow_grad/Reshape*
_output_shapes
: 
№
6gradients/conv2d_2/pow_grad/tuple/control_dependency_1Identity!gradients/conv2d_2/pow_grad/mul_3-^gradients/conv2d_2/pow_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_2/pow_grad/mul_3*'
_output_shapes
:1
А
┤
)gradients/activation_1/Relu_grad/ReluGradReluGrad0gradients/max_pooling2d/MaxPool_grad/MaxPoolGradactivation_1/Relu*
T0*/
_output_shapes
:         
К
1gradients/conv2d_3/add_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
╕
gradients/conv2d_3/add_grad/SumSumgradients/conv2d_3/Log_grad/mul1gradients/conv2d_3/add_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
l
)gradients/conv2d_3/add_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
й
#gradients/conv2d_3/add_grad/ReshapeReshapegradients/conv2d_3/add_grad/Sum)gradients/conv2d_3/add_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
|
,gradients/conv2d_3/add_grad/tuple/group_depsNoOp ^gradients/conv2d_3/Log_grad/mul$^gradients/conv2d_3/add_grad/Reshape
ў
4gradients/conv2d_3/add_grad/tuple/control_dependencyIdentitygradients/conv2d_3/Log_grad/mul-^gradients/conv2d_3/add_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/conv2d_3/Log_grad/mul*(
_output_shapes
:АА
я
6gradients/conv2d_3/add_grad/tuple/control_dependency_1Identity#gradients/conv2d_3/add_grad/Reshape-^gradients/conv2d_3/add_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_3/add_grad/Reshape*
_output_shapes
: 
К
1gradients/conv2d_4/add_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
╕
gradients/conv2d_4/add_grad/SumSumgradients/conv2d_4/Log_grad/mul1gradients/conv2d_4/add_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
l
)gradients/conv2d_4/add_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
й
#gradients/conv2d_4/add_grad/ReshapeReshapegradients/conv2d_4/add_grad/Sum)gradients/conv2d_4/add_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
|
,gradients/conv2d_4/add_grad/tuple/group_depsNoOp ^gradients/conv2d_4/Log_grad/mul$^gradients/conv2d_4/add_grad/Reshape
Ў
4gradients/conv2d_4/add_grad/tuple/control_dependencyIdentitygradients/conv2d_4/Log_grad/mul-^gradients/conv2d_4/add_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/conv2d_4/Log_grad/mul*'
_output_shapes
:А

я
6gradients/conv2d_4/add_grad/tuple/control_dependency_1Identity#gradients/conv2d_4/add_grad/Reshape-^gradients/conv2d_4/add_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_4/add_grad/Reshape*
_output_shapes
: 
~
&gradients/flatten/Reshape_1_grad/ShapeShapemax_pooling2d_1/Identity*
T0*
_output_shapes
:*
out_type0
═
(gradients/flatten/Reshape_1_grad/ReshapeReshape(gradients/reshape/Reshape_2_grad/Reshape&gradients/flatten/Reshape_1_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         

Ь
#gradients/conv2d/Conv2D_grad/ShapeNShapeNPlaceholderconv2d/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::*
out_type0
Ў
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/Conv2D/ReadVariableOp)gradients/activation_1/Relu_grad/ReluGrad*
T0*/
_output_shapes
:         *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
р
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder%gradients/conv2d/Conv2D_grad/ShapeN:1)gradients/activation_1/Relu_grad/ReluGrad*
T0*&
_output_shapes
:*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
Ь
-gradients/conv2d/Conv2D_grad/tuple/group_depsNoOp2^gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter1^gradients/conv2d/Conv2D_grad/Conv2DBackpropInput
в
5gradients/conv2d/Conv2D_grad/tuple/control_dependencyIdentity0gradients/conv2d/Conv2D_grad/Conv2DBackpropInput.^gradients/conv2d/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         
Э
7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1Identity1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter.^gradients/conv2d/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
x
 gradients/conv2d_3/Abs_grad/SignSignconv2d_3/Abs/ReadVariableOp*
T0*(
_output_shapes
:АА
▒
gradients/conv2d_3/Abs_grad/mulMul4gradients/conv2d_3/add_grad/tuple/control_dependency gradients/conv2d_3/Abs_grad/Sign*
T0*(
_output_shapes
:АА
w
 gradients/conv2d_4/Abs_grad/SignSignconv2d_4/Abs/ReadVariableOp*
T0*'
_output_shapes
:А

░
gradients/conv2d_4/Abs_grad/mulMul4gradients/conv2d_4/add_grad/tuple/control_dependency gradients/conv2d_4/Abs_grad/Sign*
T0*'
_output_shapes
:А

М
3gradients/conv2d_2/add_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
╙
!gradients/conv2d_2/add_1_grad/SumSum6gradients/conv2d_2/pow_grad/tuple/control_dependency_13gradients/conv2d_2/add_1_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
n
+gradients/conv2d_2/add_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
п
%gradients/conv2d_2/add_1_grad/ReshapeReshape!gradients/conv2d_2/add_1_grad/Sum+gradients/conv2d_2/add_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
Ч
.gradients/conv2d_2/add_1_grad/tuple/group_depsNoOp&^gradients/conv2d_2/add_1_grad/Reshape7^gradients/conv2d_2/pow_grad/tuple/control_dependency_1
ї
6gradients/conv2d_2/add_1_grad/tuple/control_dependencyIdentity%gradients/conv2d_2/add_1_grad/Reshape/^gradients/conv2d_2/add_1_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/conv2d_2/add_1_grad/Reshape*
_output_shapes
: 
Х
8gradients/conv2d_2/add_1_grad/tuple/control_dependency_1Identity6gradients/conv2d_2/pow_grad/tuple/control_dependency_1/^gradients/conv2d_2/add_1_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_2/pow_grad/mul_3*'
_output_shapes
:1
А
л
4gradients/max_pooling2d_1/MaxPool_1_grad/MaxPoolGradMaxPoolGradactivation_2/Identitymax_pooling2d_1/MaxPool_1(gradients/flatten/Reshape_1_grad/Reshape*
T0*/
_output_shapes
:         
*
data_formatNHWC*
ksize
*
paddingSAME*
strides

д
gradients/conv2d_2/mul_grad/MulMul8gradients/conv2d_2/add_1_grad/tuple/control_dependency_1conv2d_2/truediv*
T0*'
_output_shapes
:1
А
К
1gradients/conv2d_2/mul_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
╕
gradients/conv2d_2/mul_grad/SumSumgradients/conv2d_2/mul_grad/Mul1gradients/conv2d_2/mul_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
l
)gradients/conv2d_2/mul_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
й
#gradients/conv2d_2/mul_grad/ReshapeReshapegradients/conv2d_2/mul_grad/Sum)gradients/conv2d_2/mul_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
н
!gradients/conv2d_2/mul_grad/Mul_1Mulconv2d_2/ReadVariableOp8gradients/conv2d_2/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:1
А
~
,gradients/conv2d_2/mul_grad/tuple/group_depsNoOp"^gradients/conv2d_2/mul_grad/Mul_1$^gradients/conv2d_2/mul_grad/Reshape
э
4gradients/conv2d_2/mul_grad/tuple/control_dependencyIdentity#gradients/conv2d_2/mul_grad/Reshape-^gradients/conv2d_2/mul_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_2/mul_grad/Reshape*
_output_shapes
: 
№
6gradients/conv2d_2/mul_grad/tuple/control_dependency_1Identity!gradients/conv2d_2/mul_grad/Mul_1-^gradients/conv2d_2/mul_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_2/mul_grad/Mul_1*'
_output_shapes
:1
А
■
gradients/AddN_11AddNgradients/L2Loss_3_grad/mul9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1gradients/conv2d_3/Abs_grad/mul*
N*
T0*.
_class$
" loc:@gradients/L2Loss_3_grad/mul*(
_output_shapes
:АА
¤
gradients/AddN_12AddNgradients/L2Loss_4_grad/mul9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1gradients/conv2d_4/Abs_grad/mul*
N*
T0*.
_class$
" loc:@gradients/L2Loss_4_grad/mul*'
_output_shapes
:А

Н
gradients/AddN_13AddN.gradients/add_17_grad/tuple/control_dependency.gradients/add_15_grad/tuple/control_dependency6gradients/conv2d_2/add_1_grad/tuple/control_dependency*
N*
T0*0
_class&
$"loc:@gradients/add_17_grad/Reshape*
_output_shapes
: 
~
%gradients/conv2d_2/truediv_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
j
'gradients/conv2d_2/truediv_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
█
5gradients/conv2d_2/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/conv2d_2/truediv_grad/Shape'gradients/conv2d_2/truediv_grad/Shape_1*
T0*2
_output_shapes 
:         :         
м
'gradients/conv2d_2/truediv_grad/RealDivRealDiv6gradients/conv2d_2/mul_grad/tuple/control_dependency_1conv2d_2/Log_1*
T0*'
_output_shapes
:1
А
╒
#gradients/conv2d_2/truediv_grad/SumSum'gradients/conv2d_2/truediv_grad/RealDiv5gradients/conv2d_2/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0*#
_output_shapes
:1
А*
	keep_dims( 
╛
'gradients/conv2d_2/truediv_grad/ReshapeReshape#gradients/conv2d_2/truediv_grad/Sum%gradients/conv2d_2/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:1
А
j
#gradients/conv2d_2/truediv_grad/NegNegconv2d_2/Log*
T0*'
_output_shapes
:1
А
Ы
)gradients/conv2d_2/truediv_grad/RealDiv_1RealDiv#gradients/conv2d_2/truediv_grad/Negconv2d_2/Log_1*
T0*'
_output_shapes
:1
А
б
)gradients/conv2d_2/truediv_grad/RealDiv_2RealDiv)gradients/conv2d_2/truediv_grad/RealDiv_1conv2d_2/Log_1*
T0*'
_output_shapes
:1
А
┐
#gradients/conv2d_2/truediv_grad/mulMul6gradients/conv2d_2/mul_grad/tuple/control_dependency_1)gradients/conv2d_2/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:1
А
╚
%gradients/conv2d_2/truediv_grad/Sum_1Sum#gradients/conv2d_2/truediv_grad/mul7gradients/conv2d_2/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
│
)gradients/conv2d_2/truediv_grad/Reshape_1Reshape%gradients/conv2d_2/truediv_grad/Sum_1'gradients/conv2d_2/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
О
0gradients/conv2d_2/truediv_grad/tuple/group_depsNoOp(^gradients/conv2d_2/truediv_grad/Reshape*^gradients/conv2d_2/truediv_grad/Reshape_1
О
8gradients/conv2d_2/truediv_grad/tuple/control_dependencyIdentity'gradients/conv2d_2/truediv_grad/Reshape1^gradients/conv2d_2/truediv_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/conv2d_2/truediv_grad/Reshape*'
_output_shapes
:1
А
Г
:gradients/conv2d_2/truediv_grad/tuple/control_dependency_1Identity)gradients/conv2d_2/truediv_grad/Reshape_11^gradients/conv2d_2/truediv_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/conv2d_2/truediv_grad/Reshape_1*
_output_shapes
: 
╝
+gradients/activation_2/Relu_1_grad/ReluGradReluGrad4gradients/max_pooling2d_1/MaxPool_1_grad/MaxPoolGradactivation_2/Relu_1*
T0*/
_output_shapes
:         

И
gradients/AddN_14AddN-gradients/mul_5_grad/tuple/control_dependency-gradients/mul_4_grad/tuple/control_dependency4gradients/conv2d_2/mul_grad/tuple/control_dependency*
N*
T0*/
_class%
#!loc:@gradients/mul_5_grad/Reshape*
_output_shapes
: 
п
&gradients/conv2d_2/Log_grad/Reciprocal
Reciprocalconv2d_2/add9^gradients/conv2d_2/truediv_grad/tuple/control_dependency*
T0*'
_output_shapes
:1
А
║
gradients/conv2d_2/Log_grad/mulMul8gradients/conv2d_2/truediv_grad/tuple/control_dependency&gradients/conv2d_2/Log_grad/Reciprocal*
T0*'
_output_shapes
:1
А
Э
'gradients/conv2d_1/Conv2D_1_grad/ShapeNShapeNmax_pooling2d/Identityconv2d_1/mul_2*
N*
T0* 
_output_shapes
::*
out_type0
Є
4gradients/conv2d_1/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/conv2d_1/Conv2D_1_grad/ShapeNconv2d_1/mul_2+gradients/activation_2/Relu_1_grad/ReluGrad*
T0*/
_output_shapes
:         *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
ї
5gradients/conv2d_1/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/Identity)gradients/conv2d_1/Conv2D_1_grad/ShapeN:1+gradients/activation_2/Relu_1_grad/ReluGrad*
T0*&
_output_shapes
:
*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
и
1gradients/conv2d_1/Conv2D_1_grad/tuple/group_depsNoOp6^gradients/conv2d_1/Conv2D_1_grad/Conv2DBackpropFilter5^gradients/conv2d_1/Conv2D_1_grad/Conv2DBackpropInput
▓
9gradients/conv2d_1/Conv2D_1_grad/tuple/control_dependencyIdentity4gradients/conv2d_1/Conv2D_1_grad/Conv2DBackpropInput2^gradients/conv2d_1/Conv2D_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/conv2d_1/Conv2D_1_grad/Conv2DBackpropInput*/
_output_shapes
:         
н
;gradients/conv2d_1/Conv2D_1_grad/tuple/control_dependency_1Identity5gradients/conv2d_1/Conv2D_1_grad/Conv2DBackpropFilter2^gradients/conv2d_1/Conv2D_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/conv2d_1/Conv2D_1_grad/Conv2DBackpropFilter*&
_output_shapes
:

К
1gradients/conv2d_2/add_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
╕
gradients/conv2d_2/add_grad/SumSumgradients/conv2d_2/Log_grad/mul1gradients/conv2d_2/add_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
l
)gradients/conv2d_2/add_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
й
#gradients/conv2d_2/add_grad/ReshapeReshapegradients/conv2d_2/add_grad/Sum)gradients/conv2d_2/add_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
|
,gradients/conv2d_2/add_grad/tuple/group_depsNoOp ^gradients/conv2d_2/Log_grad/mul$^gradients/conv2d_2/add_grad/Reshape
Ў
4gradients/conv2d_2/add_grad/tuple/control_dependencyIdentitygradients/conv2d_2/Log_grad/mul-^gradients/conv2d_2/add_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/conv2d_2/Log_grad/mul*'
_output_shapes
:1
А
я
6gradients/conv2d_2/add_grad/tuple/control_dependency_1Identity#gradients/conv2d_2/add_grad/Reshape-^gradients/conv2d_2/add_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_2/add_grad/Reshape*
_output_shapes
: 
ё
gradients/AddN_15AddNgradients/Abs_4_grad/mulgradients/Abs_3_grad/mul;gradients/conv2d_1/Conv2D_1_grad/tuple/control_dependency_1*
N*
T0*+
_class!
loc:@gradients/Abs_4_grad/mul*&
_output_shapes
:

z
!gradients/conv2d_1/mul_2_grad/MulMulgradients/AddN_15conv2d_1/pow*
T0*&
_output_shapes
:

Г
#gradients/conv2d_1/mul_2_grad/Mul_1Mulgradients/AddN_15conv2d_1/lp_weights*
T0*&
_output_shapes
:

А
.gradients/conv2d_1/mul_2_grad/tuple/group_depsNoOp"^gradients/conv2d_1/mul_2_grad/Mul$^gradients/conv2d_1/mul_2_grad/Mul_1
¤
6gradients/conv2d_1/mul_2_grad/tuple/control_dependencyIdentity!gradients/conv2d_1/mul_2_grad/Mul/^gradients/conv2d_1/mul_2_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_1/mul_2_grad/Mul*&
_output_shapes
:

Г
8gradients/conv2d_1/mul_2_grad/tuple/control_dependency_1Identity#gradients/conv2d_1/mul_2_grad/Mul_1/^gradients/conv2d_1/mul_2_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_1/mul_2_grad/Mul_1*&
_output_shapes
:

w
 gradients/conv2d_2/Abs_grad/SignSignconv2d_2/Abs/ReadVariableOp*
T0*'
_output_shapes
:1
А
░
gradients/conv2d_2/Abs_grad/mulMul4gradients/conv2d_2/add_grad/tuple/control_dependency gradients/conv2d_2/Abs_grad/Sign*
T0*'
_output_shapes
:1
А
╕
2gradients/max_pooling2d/MaxPool_1_grad/MaxPoolGradMaxPoolGradactivation_1/Identitymax_pooling2d/MaxPool_19gradients/conv2d_1/Conv2D_1_grad/tuple/control_dependency*
T0*/
_output_shapes
:         *
data_formatNHWC*
ksize
*
paddingSAME*
strides

░
gradients/conv2d_1/pow_grad/mulMul8gradients/conv2d_1/mul_2_grad/tuple/control_dependency_1conv2d_1/differentiable_round*
T0*&
_output_shapes
:

f
!gradients/conv2d_1/pow_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Щ
gradients/conv2d_1/pow_grad/subSubconv2d_1/differentiable_round!gradients/conv2d_1/pow_grad/sub/y*
T0*&
_output_shapes
:

И
gradients/conv2d_1/pow_grad/PowPowconv2d_1/pow/xgradients/conv2d_1/pow_grad/sub*
T0*&
_output_shapes
:

Ы
!gradients/conv2d_1/pow_grad/mul_1Mulgradients/conv2d_1/pow_grad/mulgradients/conv2d_1/pow_grad/Pow*
T0*&
_output_shapes
:

К
1gradients/conv2d_1/pow_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
║
gradients/conv2d_1/pow_grad/SumSum!gradients/conv2d_1/pow_grad/mul_11gradients/conv2d_1/pow_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
l
)gradients/conv2d_1/pow_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
й
#gradients/conv2d_1/pow_grad/ReshapeReshapegradients/conv2d_1/pow_grad/Sum)gradients/conv2d_1/pow_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
j
%gradients/conv2d_1/pow_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
Ж
#gradients/conv2d_1/pow_grad/GreaterGreaterconv2d_1/pow/x%gradients/conv2d_1/pow_grad/Greater/y*
T0*
_output_shapes
: 
n
+gradients/conv2d_1/pow_grad/ones_like/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
p
+gradients/conv2d_1/pow_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
║
%gradients/conv2d_1/pow_grad/ones_likeFill+gradients/conv2d_1/pow_grad/ones_like/Shape+gradients/conv2d_1/pow_grad/ones_like/Const*
T0*
_output_shapes
: *

index_type0
й
"gradients/conv2d_1/pow_grad/SelectSelect#gradients/conv2d_1/pow_grad/Greaterconv2d_1/pow/x%gradients/conv2d_1/pow_grad/ones_like*
T0*
_output_shapes
: 
k
gradients/conv2d_1/pow_grad/LogLog"gradients/conv2d_1/pow_grad/Select*
T0*
_output_shapes
: 
k
&gradients/conv2d_1/pow_grad/zeros_likeConst*
_output_shapes
: *
dtype0*
valueB
 *    
╜
$gradients/conv2d_1/pow_grad/Select_1Select#gradients/conv2d_1/pow_grad/Greatergradients/conv2d_1/pow_grad/Log&gradients/conv2d_1/pow_grad/zeros_like*
T0*
_output_shapes
: 
б
!gradients/conv2d_1/pow_grad/mul_2Mul8gradients/conv2d_1/mul_2_grad/tuple/control_dependency_1conv2d_1/pow*
T0*&
_output_shapes
:

в
!gradients/conv2d_1/pow_grad/mul_3Mul!gradients/conv2d_1/pow_grad/mul_2$gradients/conv2d_1/pow_grad/Select_1*
T0*&
_output_shapes
:

~
,gradients/conv2d_1/pow_grad/tuple/group_depsNoOp$^gradients/conv2d_1/pow_grad/Reshape"^gradients/conv2d_1/pow_grad/mul_3
э
4gradients/conv2d_1/pow_grad/tuple/control_dependencyIdentity#gradients/conv2d_1/pow_grad/Reshape-^gradients/conv2d_1/pow_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_1/pow_grad/Reshape*
_output_shapes
: 
√
6gradients/conv2d_1/pow_grad/tuple/control_dependency_1Identity!gradients/conv2d_1/pow_grad/mul_3-^gradients/conv2d_1/pow_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_1/pow_grad/mul_3*&
_output_shapes
:

¤
gradients/AddN_16AddNgradients/L2Loss_2_grad/mul9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1gradients/conv2d_2/Abs_grad/mul*
N*
T0*.
_class$
" loc:@gradients/L2Loss_2_grad/mul*'
_output_shapes
:1
А
║
+gradients/activation_1/Relu_1_grad/ReluGradReluGrad2gradients/max_pooling2d/MaxPool_1_grad/MaxPoolGradactivation_1/Relu_1*
T0*/
_output_shapes
:         
М
3gradients/conv2d_1/add_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
╙
!gradients/conv2d_1/add_1_grad/SumSum6gradients/conv2d_1/pow_grad/tuple/control_dependency_13gradients/conv2d_1/add_1_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
n
+gradients/conv2d_1/add_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
п
%gradients/conv2d_1/add_1_grad/ReshapeReshape!gradients/conv2d_1/add_1_grad/Sum+gradients/conv2d_1/add_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
Ч
.gradients/conv2d_1/add_1_grad/tuple/group_depsNoOp&^gradients/conv2d_1/add_1_grad/Reshape7^gradients/conv2d_1/pow_grad/tuple/control_dependency_1
ї
6gradients/conv2d_1/add_1_grad/tuple/control_dependencyIdentity%gradients/conv2d_1/add_1_grad/Reshape/^gradients/conv2d_1/add_1_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/conv2d_1/add_1_grad/Reshape*
_output_shapes
: 
Ф
8gradients/conv2d_1/add_1_grad/tuple/control_dependency_1Identity6gradients/conv2d_1/pow_grad/tuple/control_dependency_1/^gradients/conv2d_1/add_1_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_1/pow_grad/mul_3*&
_output_shapes
:

Ц
%gradients/conv2d/Conv2D_1_grad/ShapeNShapeNactivation/Identityconv2d/mul_2*
N*
T0* 
_output_shapes
::*
out_type0
ь
2gradients/conv2d/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d/Conv2D_1_grad/ShapeNconv2d/mul_2+gradients/activation_1/Relu_1_grad/ReluGrad*
T0*/
_output_shapes
:         *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
ю
3gradients/conv2d/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilteractivation/Identity'gradients/conv2d/Conv2D_1_grad/ShapeN:1+gradients/activation_1/Relu_1_grad/ReluGrad*
T0*&
_output_shapes
:*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
в
/gradients/conv2d/Conv2D_1_grad/tuple/group_depsNoOp4^gradients/conv2d/Conv2D_1_grad/Conv2DBackpropFilter3^gradients/conv2d/Conv2D_1_grad/Conv2DBackpropInput
к
7gradients/conv2d/Conv2D_1_grad/tuple/control_dependencyIdentity2gradients/conv2d/Conv2D_1_grad/Conv2DBackpropInput0^gradients/conv2d/Conv2D_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d/Conv2D_1_grad/Conv2DBackpropInput*/
_output_shapes
:         
е
9gradients/conv2d/Conv2D_1_grad/tuple/control_dependency_1Identity3gradients/conv2d/Conv2D_1_grad/Conv2DBackpropFilter0^gradients/conv2d/Conv2D_1_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv2d/Conv2D_1_grad/Conv2DBackpropFilter*&
_output_shapes
:
г
gradients/conv2d_1/mul_grad/MulMul8gradients/conv2d_1/add_1_grad/tuple/control_dependency_1conv2d_1/truediv*
T0*&
_output_shapes
:

К
1gradients/conv2d_1/mul_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
╕
gradients/conv2d_1/mul_grad/SumSumgradients/conv2d_1/mul_grad/Mul1gradients/conv2d_1/mul_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
l
)gradients/conv2d_1/mul_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
й
#gradients/conv2d_1/mul_grad/ReshapeReshapegradients/conv2d_1/mul_grad/Sum)gradients/conv2d_1/mul_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
м
!gradients/conv2d_1/mul_grad/Mul_1Mulconv2d_1/ReadVariableOp8gradients/conv2d_1/add_1_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:

~
,gradients/conv2d_1/mul_grad/tuple/group_depsNoOp"^gradients/conv2d_1/mul_grad/Mul_1$^gradients/conv2d_1/mul_grad/Reshape
э
4gradients/conv2d_1/mul_grad/tuple/control_dependencyIdentity#gradients/conv2d_1/mul_grad/Reshape-^gradients/conv2d_1/mul_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_1/mul_grad/Reshape*
_output_shapes
: 
√
6gradients/conv2d_1/mul_grad/tuple/control_dependency_1Identity!gradients/conv2d_1/mul_grad/Mul_1-^gradients/conv2d_1/mul_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d_1/mul_grad/Mul_1*&
_output_shapes
:

э
gradients/AddN_17AddNgradients/Abs_1_grad/mulgradients/Abs_grad/mul9gradients/conv2d/Conv2D_1_grad/tuple/control_dependency_1*
N*
T0*+
_class!
loc:@gradients/Abs_1_grad/mul*&
_output_shapes
:
v
gradients/conv2d/mul_2_grad/MulMulgradients/AddN_17
conv2d/pow*
T0*&
_output_shapes
:

!gradients/conv2d/mul_2_grad/Mul_1Mulgradients/AddN_17conv2d/lp_weights*
T0*&
_output_shapes
:
z
,gradients/conv2d/mul_2_grad/tuple/group_depsNoOp ^gradients/conv2d/mul_2_grad/Mul"^gradients/conv2d/mul_2_grad/Mul_1
ї
4gradients/conv2d/mul_2_grad/tuple/control_dependencyIdentitygradients/conv2d/mul_2_grad/Mul-^gradients/conv2d/mul_2_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/conv2d/mul_2_grad/Mul*&
_output_shapes
:
√
6gradients/conv2d/mul_2_grad/tuple/control_dependency_1Identity!gradients/conv2d/mul_2_grad/Mul_1-^gradients/conv2d/mul_2_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d/mul_2_grad/Mul_1*&
_output_shapes
:
М
gradients/AddN_18AddN.gradients/add_10_grad/tuple/control_dependency-gradients/add_8_grad/tuple/control_dependency6gradients/conv2d_1/add_1_grad/tuple/control_dependency*
N*
T0*0
_class&
$"loc:@gradients/add_10_grad/Reshape*
_output_shapes
: 
~
%gradients/conv2d_1/truediv_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
j
'gradients/conv2d_1/truediv_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
█
5gradients/conv2d_1/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/conv2d_1/truediv_grad/Shape'gradients/conv2d_1/truediv_grad/Shape_1*
T0*2
_output_shapes 
:         :         
л
'gradients/conv2d_1/truediv_grad/RealDivRealDiv6gradients/conv2d_1/mul_grad/tuple/control_dependency_1conv2d_1/Log_1*
T0*&
_output_shapes
:

╪
#gradients/conv2d_1/truediv_grad/SumSum'gradients/conv2d_1/truediv_grad/RealDiv5gradients/conv2d_1/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0*&
_output_shapes
:
*
	keep_dims( 
╜
'gradients/conv2d_1/truediv_grad/ReshapeReshape#gradients/conv2d_1/truediv_grad/Sum%gradients/conv2d_1/truediv_grad/Shape*
T0*
Tshape0*&
_output_shapes
:

i
#gradients/conv2d_1/truediv_grad/NegNegconv2d_1/Log*
T0*&
_output_shapes
:

Ъ
)gradients/conv2d_1/truediv_grad/RealDiv_1RealDiv#gradients/conv2d_1/truediv_grad/Negconv2d_1/Log_1*
T0*&
_output_shapes
:

а
)gradients/conv2d_1/truediv_grad/RealDiv_2RealDiv)gradients/conv2d_1/truediv_grad/RealDiv_1conv2d_1/Log_1*
T0*&
_output_shapes
:

╛
#gradients/conv2d_1/truediv_grad/mulMul6gradients/conv2d_1/mul_grad/tuple/control_dependency_1)gradients/conv2d_1/truediv_grad/RealDiv_2*
T0*&
_output_shapes
:

╚
%gradients/conv2d_1/truediv_grad/Sum_1Sum#gradients/conv2d_1/truediv_grad/mul7gradients/conv2d_1/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
│
)gradients/conv2d_1/truediv_grad/Reshape_1Reshape%gradients/conv2d_1/truediv_grad/Sum_1'gradients/conv2d_1/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
О
0gradients/conv2d_1/truediv_grad/tuple/group_depsNoOp(^gradients/conv2d_1/truediv_grad/Reshape*^gradients/conv2d_1/truediv_grad/Reshape_1
Н
8gradients/conv2d_1/truediv_grad/tuple/control_dependencyIdentity'gradients/conv2d_1/truediv_grad/Reshape1^gradients/conv2d_1/truediv_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/conv2d_1/truediv_grad/Reshape*&
_output_shapes
:

Г
:gradients/conv2d_1/truediv_grad/tuple/control_dependency_1Identity)gradients/conv2d_1/truediv_grad/Reshape_11^gradients/conv2d_1/truediv_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/conv2d_1/truediv_grad/Reshape_1*
_output_shapes
: 
к
gradients/conv2d/pow_grad/mulMul6gradients/conv2d/mul_2_grad/tuple/control_dependency_1conv2d/differentiable_round*
T0*&
_output_shapes
:
d
gradients/conv2d/pow_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
У
gradients/conv2d/pow_grad/subSubconv2d/differentiable_roundgradients/conv2d/pow_grad/sub/y*
T0*&
_output_shapes
:
В
gradients/conv2d/pow_grad/PowPowconv2d/pow/xgradients/conv2d/pow_grad/sub*
T0*&
_output_shapes
:
Х
gradients/conv2d/pow_grad/mul_1Mulgradients/conv2d/pow_grad/mulgradients/conv2d/pow_grad/Pow*
T0*&
_output_shapes
:
И
/gradients/conv2d/pow_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
┤
gradients/conv2d/pow_grad/SumSumgradients/conv2d/pow_grad/mul_1/gradients/conv2d/pow_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
j
'gradients/conv2d/pow_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
г
!gradients/conv2d/pow_grad/ReshapeReshapegradients/conv2d/pow_grad/Sum'gradients/conv2d/pow_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
h
#gradients/conv2d/pow_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
А
!gradients/conv2d/pow_grad/GreaterGreaterconv2d/pow/x#gradients/conv2d/pow_grad/Greater/y*
T0*
_output_shapes
: 
l
)gradients/conv2d/pow_grad/ones_like/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
n
)gradients/conv2d/pow_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
┤
#gradients/conv2d/pow_grad/ones_likeFill)gradients/conv2d/pow_grad/ones_like/Shape)gradients/conv2d/pow_grad/ones_like/Const*
T0*
_output_shapes
: *

index_type0
б
 gradients/conv2d/pow_grad/SelectSelect!gradients/conv2d/pow_grad/Greaterconv2d/pow/x#gradients/conv2d/pow_grad/ones_like*
T0*
_output_shapes
: 
g
gradients/conv2d/pow_grad/LogLog gradients/conv2d/pow_grad/Select*
T0*
_output_shapes
: 
i
$gradients/conv2d/pow_grad/zeros_likeConst*
_output_shapes
: *
dtype0*
valueB
 *    
╡
"gradients/conv2d/pow_grad/Select_1Select!gradients/conv2d/pow_grad/Greatergradients/conv2d/pow_grad/Log$gradients/conv2d/pow_grad/zeros_like*
T0*
_output_shapes
: 
Ы
gradients/conv2d/pow_grad/mul_2Mul6gradients/conv2d/mul_2_grad/tuple/control_dependency_1
conv2d/pow*
T0*&
_output_shapes
:
Ь
gradients/conv2d/pow_grad/mul_3Mulgradients/conv2d/pow_grad/mul_2"gradients/conv2d/pow_grad/Select_1*
T0*&
_output_shapes
:
x
*gradients/conv2d/pow_grad/tuple/group_depsNoOp"^gradients/conv2d/pow_grad/Reshape ^gradients/conv2d/pow_grad/mul_3
х
2gradients/conv2d/pow_grad/tuple/control_dependencyIdentity!gradients/conv2d/pow_grad/Reshape+^gradients/conv2d/pow_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d/pow_grad/Reshape*
_output_shapes
: 
є
4gradients/conv2d/pow_grad/tuple/control_dependency_1Identitygradients/conv2d/pow_grad/mul_3+^gradients/conv2d/pow_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/conv2d/pow_grad/mul_3*&
_output_shapes
:
И
gradients/AddN_19AddN-gradients/mul_3_grad/tuple/control_dependency-gradients/mul_2_grad/tuple/control_dependency4gradients/conv2d_1/mul_grad/tuple/control_dependency*
N*
T0*/
_class%
#!loc:@gradients/mul_3_grad/Reshape*
_output_shapes
: 
о
&gradients/conv2d_1/Log_grad/Reciprocal
Reciprocalconv2d_1/add9^gradients/conv2d_1/truediv_grad/tuple/control_dependency*
T0*&
_output_shapes
:

╣
gradients/conv2d_1/Log_grad/mulMul8gradients/conv2d_1/truediv_grad/tuple/control_dependency&gradients/conv2d_1/Log_grad/Reciprocal*
T0*&
_output_shapes
:

К
1gradients/conv2d_1/add_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
╕
gradients/conv2d_1/add_grad/SumSumgradients/conv2d_1/Log_grad/mul1gradients/conv2d_1/add_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
l
)gradients/conv2d_1/add_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
й
#gradients/conv2d_1/add_grad/ReshapeReshapegradients/conv2d_1/add_grad/Sum)gradients/conv2d_1/add_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
|
,gradients/conv2d_1/add_grad/tuple/group_depsNoOp ^gradients/conv2d_1/Log_grad/mul$^gradients/conv2d_1/add_grad/Reshape
ї
4gradients/conv2d_1/add_grad/tuple/control_dependencyIdentitygradients/conv2d_1/Log_grad/mul-^gradients/conv2d_1/add_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/conv2d_1/Log_grad/mul*&
_output_shapes
:

я
6gradients/conv2d_1/add_grad/tuple/control_dependency_1Identity#gradients/conv2d_1/add_grad/Reshape-^gradients/conv2d_1/add_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d_1/add_grad/Reshape*
_output_shapes
: 
К
1gradients/conv2d/add_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
═
gradients/conv2d/add_1_grad/SumSum4gradients/conv2d/pow_grad/tuple/control_dependency_11gradients/conv2d/add_1_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
l
)gradients/conv2d/add_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
й
#gradients/conv2d/add_1_grad/ReshapeReshapegradients/conv2d/add_1_grad/Sum)gradients/conv2d/add_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
С
,gradients/conv2d/add_1_grad/tuple/group_depsNoOp$^gradients/conv2d/add_1_grad/Reshape5^gradients/conv2d/pow_grad/tuple/control_dependency_1
э
4gradients/conv2d/add_1_grad/tuple/control_dependencyIdentity#gradients/conv2d/add_1_grad/Reshape-^gradients/conv2d/add_1_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d/add_1_grad/Reshape*
_output_shapes
: 
М
6gradients/conv2d/add_1_grad/tuple/control_dependency_1Identity4gradients/conv2d/pow_grad/tuple/control_dependency_1-^gradients/conv2d/add_1_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/conv2d/pow_grad/mul_3*&
_output_shapes
:
v
 gradients/conv2d_1/Abs_grad/SignSignconv2d_1/Abs/ReadVariableOp*
T0*&
_output_shapes
:

п
gradients/conv2d_1/Abs_grad/mulMul4gradients/conv2d_1/add_grad/tuple/control_dependency gradients/conv2d_1/Abs_grad/Sign*
T0*&
_output_shapes
:

Э
gradients/conv2d/mul_grad/MulMul6gradients/conv2d/add_1_grad/tuple/control_dependency_1conv2d/truediv*
T0*&
_output_shapes
:
И
/gradients/conv2d/mul_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
▓
gradients/conv2d/mul_grad/SumSumgradients/conv2d/mul_grad/Mul/gradients/conv2d/mul_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
j
'gradients/conv2d/mul_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
г
!gradients/conv2d/mul_grad/ReshapeReshapegradients/conv2d/mul_grad/Sum'gradients/conv2d/mul_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
ж
gradients/conv2d/mul_grad/Mul_1Mulconv2d/ReadVariableOp6gradients/conv2d/add_1_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:
x
*gradients/conv2d/mul_grad/tuple/group_depsNoOp ^gradients/conv2d/mul_grad/Mul_1"^gradients/conv2d/mul_grad/Reshape
х
2gradients/conv2d/mul_grad/tuple/control_dependencyIdentity!gradients/conv2d/mul_grad/Reshape+^gradients/conv2d/mul_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d/mul_grad/Reshape*
_output_shapes
: 
є
4gradients/conv2d/mul_grad/tuple/control_dependency_1Identitygradients/conv2d/mul_grad/Mul_1+^gradients/conv2d/mul_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/conv2d/mul_grad/Mul_1*&
_output_shapes
:
И
gradients/AddN_20AddN-gradients/add_3_grad/tuple/control_dependency-gradients/add_1_grad/tuple/control_dependency4gradients/conv2d/add_1_grad/tuple/control_dependency*
N*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape*
_output_shapes
: 
|
#gradients/conv2d/truediv_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
h
%gradients/conv2d/truediv_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
╒
3gradients/conv2d/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/conv2d/truediv_grad/Shape%gradients/conv2d/truediv_grad/Shape_1*
T0*2
_output_shapes 
:         :         
е
%gradients/conv2d/truediv_grad/RealDivRealDiv4gradients/conv2d/mul_grad/tuple/control_dependency_1conv2d/Log_1*
T0*&
_output_shapes
:
╬
!gradients/conv2d/truediv_grad/SumSum%gradients/conv2d/truediv_grad/RealDiv3gradients/conv2d/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0*"
_output_shapes
:*
	keep_dims( 
╖
%gradients/conv2d/truediv_grad/ReshapeReshape!gradients/conv2d/truediv_grad/Sum#gradients/conv2d/truediv_grad/Shape*
T0*
Tshape0*&
_output_shapes
:
e
!gradients/conv2d/truediv_grad/NegNeg
conv2d/Log*
T0*&
_output_shapes
:
Ф
'gradients/conv2d/truediv_grad/RealDiv_1RealDiv!gradients/conv2d/truediv_grad/Negconv2d/Log_1*
T0*&
_output_shapes
:
Ъ
'gradients/conv2d/truediv_grad/RealDiv_2RealDiv'gradients/conv2d/truediv_grad/RealDiv_1conv2d/Log_1*
T0*&
_output_shapes
:
╕
!gradients/conv2d/truediv_grad/mulMul4gradients/conv2d/mul_grad/tuple/control_dependency_1'gradients/conv2d/truediv_grad/RealDiv_2*
T0*&
_output_shapes
:
┬
#gradients/conv2d/truediv_grad/Sum_1Sum!gradients/conv2d/truediv_grad/mul5gradients/conv2d/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
н
'gradients/conv2d/truediv_grad/Reshape_1Reshape#gradients/conv2d/truediv_grad/Sum_1%gradients/conv2d/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
И
.gradients/conv2d/truediv_grad/tuple/group_depsNoOp&^gradients/conv2d/truediv_grad/Reshape(^gradients/conv2d/truediv_grad/Reshape_1
Е
6gradients/conv2d/truediv_grad/tuple/control_dependencyIdentity%gradients/conv2d/truediv_grad/Reshape/^gradients/conv2d/truediv_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/conv2d/truediv_grad/Reshape*&
_output_shapes
:
√
8gradients/conv2d/truediv_grad/tuple/control_dependency_1Identity'gradients/conv2d/truediv_grad/Reshape_1/^gradients/conv2d/truediv_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/conv2d/truediv_grad/Reshape_1*
_output_shapes
: 
№
gradients/AddN_21AddNgradients/L2Loss_1_grad/mul9gradients/conv2d_1/Conv2D_grad/tuple/control_dependency_1gradients/conv2d_1/Abs_grad/mul*
N*
T0*.
_class$
" loc:@gradients/L2Loss_1_grad/mul*&
_output_shapes
:

Д
gradients/AddN_22AddN-gradients/mul_1_grad/tuple/control_dependency+gradients/mul_grad/tuple/control_dependency2gradients/conv2d/mul_grad/tuple/control_dependency*
N*
T0*/
_class%
#!loc:@gradients/mul_1_grad/Reshape*
_output_shapes
: 
и
$gradients/conv2d/Log_grad/Reciprocal
Reciprocal
conv2d/add7^gradients/conv2d/truediv_grad/tuple/control_dependency*
T0*&
_output_shapes
:
│
gradients/conv2d/Log_grad/mulMul6gradients/conv2d/truediv_grad/tuple/control_dependency$gradients/conv2d/Log_grad/Reciprocal*
T0*&
_output_shapes
:
И
/gradients/conv2d/add_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             
▓
gradients/conv2d/add_grad/SumSumgradients/conv2d/Log_grad/mul/gradients/conv2d/add_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
j
'gradients/conv2d/add_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
г
!gradients/conv2d/add_grad/ReshapeReshapegradients/conv2d/add_grad/Sum'gradients/conv2d/add_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
v
*gradients/conv2d/add_grad/tuple/group_depsNoOp^gradients/conv2d/Log_grad/mul"^gradients/conv2d/add_grad/Reshape
э
2gradients/conv2d/add_grad/tuple/control_dependencyIdentitygradients/conv2d/Log_grad/mul+^gradients/conv2d/add_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/conv2d/Log_grad/mul*&
_output_shapes
:
ч
4gradients/conv2d/add_grad/tuple/control_dependency_1Identity!gradients/conv2d/add_grad/Reshape+^gradients/conv2d/add_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/conv2d/add_grad/Reshape*
_output_shapes
: 
r
gradients/conv2d/Abs_grad/SignSignconv2d/Abs/ReadVariableOp*
T0*&
_output_shapes
:
й
gradients/conv2d/Abs_grad/mulMul2gradients/conv2d/add_grad/tuple/control_dependencygradients/conv2d/Abs_grad/Sign*
T0*&
_output_shapes
:
Ї
gradients/AddN_23AddNgradients/L2Loss_grad/mul7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1gradients/conv2d/Abs_grad/mul*
N*
T0*,
_class"
 loc:@gradients/L2Loss_grad/mul*&
_output_shapes
:
М
%beta1_power/Initializer/initial_valueConst* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
dtype0*
valueB
 *fff?
Э
beta1_powerVarHandleOp* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_namebeta1_power
Й
,beta1_power/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta1_power* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
g
beta1_power/AssignAssignVariableOpbeta1_power%beta1_power/Initializer/initial_value*
dtype0
Е
beta1_power/Read/ReadVariableOpReadVariableOpbeta1_power* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
dtype0
М
%beta2_power/Initializer/initial_valueConst* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
dtype0*
valueB
 *w╛?
Э
beta2_powerVarHandleOp* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_namebeta2_power
Й
,beta2_power/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta2_power* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
g
beta2_power/AssignAssignVariableOpbeta2_power%beta2_power/Initializer/initial_value*
dtype0
Е
beta2_power/Read/ReadVariableOpReadVariableOpbeta2_power* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
dtype0
Г
 intercept/Adam/Initializer/zerosConst*
_class
loc:@intercept*
_output_shapes
: *
dtype0*
valueB
 *    
Я
intercept/AdamVarHandleOp*
_class
loc:@intercept*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameintercept/Adam
Л
/intercept/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept/Adam*
_class
loc:@intercept*
_output_shapes
: 
h
intercept/Adam/AssignAssignVariableOpintercept/Adam intercept/Adam/Initializer/zeros*
dtype0
З
"intercept/Adam/Read/ReadVariableOpReadVariableOpintercept/Adam*
_class
loc:@intercept*
_output_shapes
: *
dtype0
Е
"intercept/Adam_1/Initializer/zerosConst*
_class
loc:@intercept*
_output_shapes
: *
dtype0*
valueB
 *    
г
intercept/Adam_1VarHandleOp*
_class
loc:@intercept*
_output_shapes
: *
	container *
dtype0*
shape: *!
shared_nameintercept/Adam_1
П
1intercept/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept/Adam_1*
_class
loc:@intercept*
_output_shapes
: 
n
intercept/Adam_1/AssignAssignVariableOpintercept/Adam_1"intercept/Adam_1/Initializer/zeros*
dtype0
Л
$intercept/Adam_1/Read/ReadVariableOpReadVariableOpintercept/Adam_1*
_class
loc:@intercept*
_output_shapes
: *
dtype0
{
slope/Adam/Initializer/zerosConst*
_class

loc:@slope*
_output_shapes
: *
dtype0*
valueB
 *    
У

slope/AdamVarHandleOp*
_class

loc:@slope*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_name
slope/Adam

+slope/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp
slope/Adam*
_class

loc:@slope*
_output_shapes
: 
\
slope/Adam/AssignAssignVariableOp
slope/Adamslope/Adam/Initializer/zeros*
dtype0
{
slope/Adam/Read/ReadVariableOpReadVariableOp
slope/Adam*
_class

loc:@slope*
_output_shapes
: *
dtype0
}
slope/Adam_1/Initializer/zerosConst*
_class

loc:@slope*
_output_shapes
: *
dtype0*
valueB
 *    
Ч
slope/Adam_1VarHandleOp*
_class

loc:@slope*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameslope/Adam_1
Г
-slope/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope/Adam_1*
_class

loc:@slope*
_output_shapes
: 
b
slope/Adam_1/AssignAssignVariableOpslope/Adam_1slope/Adam_1/Initializer/zeros*
dtype0

 slope/Adam_1/Read/ReadVariableOpReadVariableOpslope/Adam_1*
_class

loc:@slope*
_output_shapes
: *
dtype0
З
"intercept_1/Adam/Initializer/zerosConst*
_class
loc:@intercept_1*
_output_shapes
: *
dtype0*
valueB
 *    
е
intercept_1/AdamVarHandleOp*
_class
loc:@intercept_1*
_output_shapes
: *
	container *
dtype0*
shape: *!
shared_nameintercept_1/Adam
С
1intercept_1/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_1/Adam*
_class
loc:@intercept_1*
_output_shapes
: 
n
intercept_1/Adam/AssignAssignVariableOpintercept_1/Adam"intercept_1/Adam/Initializer/zeros*
dtype0
Н
$intercept_1/Adam/Read/ReadVariableOpReadVariableOpintercept_1/Adam*
_class
loc:@intercept_1*
_output_shapes
: *
dtype0
Й
$intercept_1/Adam_1/Initializer/zerosConst*
_class
loc:@intercept_1*
_output_shapes
: *
dtype0*
valueB
 *    
й
intercept_1/Adam_1VarHandleOp*
_class
loc:@intercept_1*
_output_shapes
: *
	container *
dtype0*
shape: *#
shared_nameintercept_1/Adam_1
Х
3intercept_1/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_1/Adam_1*
_class
loc:@intercept_1*
_output_shapes
: 
t
intercept_1/Adam_1/AssignAssignVariableOpintercept_1/Adam_1$intercept_1/Adam_1/Initializer/zeros*
dtype0
С
&intercept_1/Adam_1/Read/ReadVariableOpReadVariableOpintercept_1/Adam_1*
_class
loc:@intercept_1*
_output_shapes
: *
dtype0

slope_1/Adam/Initializer/zerosConst*
_class
loc:@slope_1*
_output_shapes
: *
dtype0*
valueB
 *    
Щ
slope_1/AdamVarHandleOp*
_class
loc:@slope_1*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameslope_1/Adam
Е
-slope_1/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_1/Adam*
_class
loc:@slope_1*
_output_shapes
: 
b
slope_1/Adam/AssignAssignVariableOpslope_1/Adamslope_1/Adam/Initializer/zeros*
dtype0
Б
 slope_1/Adam/Read/ReadVariableOpReadVariableOpslope_1/Adam*
_class
loc:@slope_1*
_output_shapes
: *
dtype0
Б
 slope_1/Adam_1/Initializer/zerosConst*
_class
loc:@slope_1*
_output_shapes
: *
dtype0*
valueB
 *    
Э
slope_1/Adam_1VarHandleOp*
_class
loc:@slope_1*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameslope_1/Adam_1
Й
/slope_1/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_1/Adam_1*
_class
loc:@slope_1*
_output_shapes
: 
h
slope_1/Adam_1/AssignAssignVariableOpslope_1/Adam_1 slope_1/Adam_1/Initializer/zeros*
dtype0
Е
"slope_1/Adam_1/Read/ReadVariableOpReadVariableOpslope_1/Adam_1*
_class
loc:@slope_1*
_output_shapes
: *
dtype0
З
"intercept_2/Adam/Initializer/zerosConst*
_class
loc:@intercept_2*
_output_shapes
: *
dtype0*
valueB
 *    
е
intercept_2/AdamVarHandleOp*
_class
loc:@intercept_2*
_output_shapes
: *
	container *
dtype0*
shape: *!
shared_nameintercept_2/Adam
С
1intercept_2/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_2/Adam*
_class
loc:@intercept_2*
_output_shapes
: 
n
intercept_2/Adam/AssignAssignVariableOpintercept_2/Adam"intercept_2/Adam/Initializer/zeros*
dtype0
Н
$intercept_2/Adam/Read/ReadVariableOpReadVariableOpintercept_2/Adam*
_class
loc:@intercept_2*
_output_shapes
: *
dtype0
Й
$intercept_2/Adam_1/Initializer/zerosConst*
_class
loc:@intercept_2*
_output_shapes
: *
dtype0*
valueB
 *    
й
intercept_2/Adam_1VarHandleOp*
_class
loc:@intercept_2*
_output_shapes
: *
	container *
dtype0*
shape: *#
shared_nameintercept_2/Adam_1
Х
3intercept_2/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_2/Adam_1*
_class
loc:@intercept_2*
_output_shapes
: 
t
intercept_2/Adam_1/AssignAssignVariableOpintercept_2/Adam_1$intercept_2/Adam_1/Initializer/zeros*
dtype0
С
&intercept_2/Adam_1/Read/ReadVariableOpReadVariableOpintercept_2/Adam_1*
_class
loc:@intercept_2*
_output_shapes
: *
dtype0

slope_2/Adam/Initializer/zerosConst*
_class
loc:@slope_2*
_output_shapes
: *
dtype0*
valueB
 *    
Щ
slope_2/AdamVarHandleOp*
_class
loc:@slope_2*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameslope_2/Adam
Е
-slope_2/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_2/Adam*
_class
loc:@slope_2*
_output_shapes
: 
b
slope_2/Adam/AssignAssignVariableOpslope_2/Adamslope_2/Adam/Initializer/zeros*
dtype0
Б
 slope_2/Adam/Read/ReadVariableOpReadVariableOpslope_2/Adam*
_class
loc:@slope_2*
_output_shapes
: *
dtype0
Б
 slope_2/Adam_1/Initializer/zerosConst*
_class
loc:@slope_2*
_output_shapes
: *
dtype0*
valueB
 *    
Э
slope_2/Adam_1VarHandleOp*
_class
loc:@slope_2*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameslope_2/Adam_1
Й
/slope_2/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_2/Adam_1*
_class
loc:@slope_2*
_output_shapes
: 
h
slope_2/Adam_1/AssignAssignVariableOpslope_2/Adam_1 slope_2/Adam_1/Initializer/zeros*
dtype0
Е
"slope_2/Adam_1/Read/ReadVariableOpReadVariableOpslope_2/Adam_1*
_class
loc:@slope_2*
_output_shapes
: *
dtype0
З
"intercept_3/Adam/Initializer/zerosConst*
_class
loc:@intercept_3*
_output_shapes
: *
dtype0*
valueB
 *    
е
intercept_3/AdamVarHandleOp*
_class
loc:@intercept_3*
_output_shapes
: *
	container *
dtype0*
shape: *!
shared_nameintercept_3/Adam
С
1intercept_3/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_3/Adam*
_class
loc:@intercept_3*
_output_shapes
: 
n
intercept_3/Adam/AssignAssignVariableOpintercept_3/Adam"intercept_3/Adam/Initializer/zeros*
dtype0
Н
$intercept_3/Adam/Read/ReadVariableOpReadVariableOpintercept_3/Adam*
_class
loc:@intercept_3*
_output_shapes
: *
dtype0
Й
$intercept_3/Adam_1/Initializer/zerosConst*
_class
loc:@intercept_3*
_output_shapes
: *
dtype0*
valueB
 *    
й
intercept_3/Adam_1VarHandleOp*
_class
loc:@intercept_3*
_output_shapes
: *
	container *
dtype0*
shape: *#
shared_nameintercept_3/Adam_1
Х
3intercept_3/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_3/Adam_1*
_class
loc:@intercept_3*
_output_shapes
: 
t
intercept_3/Adam_1/AssignAssignVariableOpintercept_3/Adam_1$intercept_3/Adam_1/Initializer/zeros*
dtype0
С
&intercept_3/Adam_1/Read/ReadVariableOpReadVariableOpintercept_3/Adam_1*
_class
loc:@intercept_3*
_output_shapes
: *
dtype0

slope_3/Adam/Initializer/zerosConst*
_class
loc:@slope_3*
_output_shapes
: *
dtype0*
valueB
 *    
Щ
slope_3/AdamVarHandleOp*
_class
loc:@slope_3*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameslope_3/Adam
Е
-slope_3/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_3/Adam*
_class
loc:@slope_3*
_output_shapes
: 
b
slope_3/Adam/AssignAssignVariableOpslope_3/Adamslope_3/Adam/Initializer/zeros*
dtype0
Б
 slope_3/Adam/Read/ReadVariableOpReadVariableOpslope_3/Adam*
_class
loc:@slope_3*
_output_shapes
: *
dtype0
Б
 slope_3/Adam_1/Initializer/zerosConst*
_class
loc:@slope_3*
_output_shapes
: *
dtype0*
valueB
 *    
Э
slope_3/Adam_1VarHandleOp*
_class
loc:@slope_3*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameslope_3/Adam_1
Й
/slope_3/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_3/Adam_1*
_class
loc:@slope_3*
_output_shapes
: 
h
slope_3/Adam_1/AssignAssignVariableOpslope_3/Adam_1 slope_3/Adam_1/Initializer/zeros*
dtype0
Е
"slope_3/Adam_1/Read/ReadVariableOpReadVariableOpslope_3/Adam_1*
_class
loc:@slope_3*
_output_shapes
: *
dtype0
З
"intercept_4/Adam/Initializer/zerosConst*
_class
loc:@intercept_4*
_output_shapes
: *
dtype0*
valueB
 *    
е
intercept_4/AdamVarHandleOp*
_class
loc:@intercept_4*
_output_shapes
: *
	container *
dtype0*
shape: *!
shared_nameintercept_4/Adam
С
1intercept_4/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_4/Adam*
_class
loc:@intercept_4*
_output_shapes
: 
n
intercept_4/Adam/AssignAssignVariableOpintercept_4/Adam"intercept_4/Adam/Initializer/zeros*
dtype0
Н
$intercept_4/Adam/Read/ReadVariableOpReadVariableOpintercept_4/Adam*
_class
loc:@intercept_4*
_output_shapes
: *
dtype0
Й
$intercept_4/Adam_1/Initializer/zerosConst*
_class
loc:@intercept_4*
_output_shapes
: *
dtype0*
valueB
 *    
й
intercept_4/Adam_1VarHandleOp*
_class
loc:@intercept_4*
_output_shapes
: *
	container *
dtype0*
shape: *#
shared_nameintercept_4/Adam_1
Х
3intercept_4/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_4/Adam_1*
_class
loc:@intercept_4*
_output_shapes
: 
t
intercept_4/Adam_1/AssignAssignVariableOpintercept_4/Adam_1$intercept_4/Adam_1/Initializer/zeros*
dtype0
С
&intercept_4/Adam_1/Read/ReadVariableOpReadVariableOpintercept_4/Adam_1*
_class
loc:@intercept_4*
_output_shapes
: *
dtype0

slope_4/Adam/Initializer/zerosConst*
_class
loc:@slope_4*
_output_shapes
: *
dtype0*
valueB
 *    
Щ
slope_4/AdamVarHandleOp*
_class
loc:@slope_4*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameslope_4/Adam
Е
-slope_4/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_4/Adam*
_class
loc:@slope_4*
_output_shapes
: 
b
slope_4/Adam/AssignAssignVariableOpslope_4/Adamslope_4/Adam/Initializer/zeros*
dtype0
Б
 slope_4/Adam/Read/ReadVariableOpReadVariableOpslope_4/Adam*
_class
loc:@slope_4*
_output_shapes
: *
dtype0
Б
 slope_4/Adam_1/Initializer/zerosConst*
_class
loc:@slope_4*
_output_shapes
: *
dtype0*
valueB
 *    
Э
slope_4/Adam_1VarHandleOp*
_class
loc:@slope_4*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameslope_4/Adam_1
Й
/slope_4/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_4/Adam_1*
_class
loc:@slope_4*
_output_shapes
: 
h
slope_4/Adam_1/AssignAssignVariableOpslope_4/Adam_1 slope_4/Adam_1/Initializer/zeros*
dtype0
Е
"slope_4/Adam_1/Read/ReadVariableOpReadVariableOpslope_4/Adam_1*
_class
loc:@slope_4*
_output_shapes
: *
dtype0
З
"intercept_5/Adam/Initializer/zerosConst*
_class
loc:@intercept_5*
_output_shapes
: *
dtype0*
valueB
 *    
е
intercept_5/AdamVarHandleOp*
_class
loc:@intercept_5*
_output_shapes
: *
	container *
dtype0*
shape: *!
shared_nameintercept_5/Adam
С
1intercept_5/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_5/Adam*
_class
loc:@intercept_5*
_output_shapes
: 
n
intercept_5/Adam/AssignAssignVariableOpintercept_5/Adam"intercept_5/Adam/Initializer/zeros*
dtype0
Н
$intercept_5/Adam/Read/ReadVariableOpReadVariableOpintercept_5/Adam*
_class
loc:@intercept_5*
_output_shapes
: *
dtype0
Й
$intercept_5/Adam_1/Initializer/zerosConst*
_class
loc:@intercept_5*
_output_shapes
: *
dtype0*
valueB
 *    
й
intercept_5/Adam_1VarHandleOp*
_class
loc:@intercept_5*
_output_shapes
: *
	container *
dtype0*
shape: *#
shared_nameintercept_5/Adam_1
Х
3intercept_5/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpintercept_5/Adam_1*
_class
loc:@intercept_5*
_output_shapes
: 
t
intercept_5/Adam_1/AssignAssignVariableOpintercept_5/Adam_1$intercept_5/Adam_1/Initializer/zeros*
dtype0
С
&intercept_5/Adam_1/Read/ReadVariableOpReadVariableOpintercept_5/Adam_1*
_class
loc:@intercept_5*
_output_shapes
: *
dtype0

slope_5/Adam/Initializer/zerosConst*
_class
loc:@slope_5*
_output_shapes
: *
dtype0*
valueB
 *    
Щ
slope_5/AdamVarHandleOp*
_class
loc:@slope_5*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameslope_5/Adam
Е
-slope_5/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_5/Adam*
_class
loc:@slope_5*
_output_shapes
: 
b
slope_5/Adam/AssignAssignVariableOpslope_5/Adamslope_5/Adam/Initializer/zeros*
dtype0
Б
 slope_5/Adam/Read/ReadVariableOpReadVariableOpslope_5/Adam*
_class
loc:@slope_5*
_output_shapes
: *
dtype0
Б
 slope_5/Adam_1/Initializer/zerosConst*
_class
loc:@slope_5*
_output_shapes
: *
dtype0*
valueB
 *    
Э
slope_5/Adam_1VarHandleOp*
_class
loc:@slope_5*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameslope_5/Adam_1
Й
/slope_5/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpslope_5/Adam_1*
_class
loc:@slope_5*
_output_shapes
: 
h
slope_5/Adam_1/AssignAssignVariableOpslope_5/Adam_1 slope_5/Adam_1/Initializer/zeros*
dtype0
Е
"slope_5/Adam_1/Read/ReadVariableOpReadVariableOpslope_5/Adam_1*
_class
loc:@slope_5*
_output_shapes
: *
dtype0
л
$conv2d/kernel/Adam/Initializer/zerosConst* 
_class
loc:@conv2d/kernel*&
_output_shapes
:*
dtype0*%
valueB*    
╗
conv2d/kernel/AdamVarHandleOp* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
	container *
dtype0*
shape:*#
shared_nameconv2d/kernel/Adam
Ч
3conv2d/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/kernel/Adam* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
t
conv2d/kernel/Adam/AssignAssignVariableOpconv2d/kernel/Adam$conv2d/kernel/Adam/Initializer/zeros*
dtype0
г
&conv2d/kernel/Adam/Read/ReadVariableOpReadVariableOpconv2d/kernel/Adam* 
_class
loc:@conv2d/kernel*&
_output_shapes
:*
dtype0
н
&conv2d/kernel/Adam_1/Initializer/zerosConst* 
_class
loc:@conv2d/kernel*&
_output_shapes
:*
dtype0*%
valueB*    
┐
conv2d/kernel/Adam_1VarHandleOp* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
	container *
dtype0*
shape:*%
shared_nameconv2d/kernel/Adam_1
Ы
5conv2d/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/kernel/Adam_1* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
z
conv2d/kernel/Adam_1/AssignAssignVariableOpconv2d/kernel/Adam_1&conv2d/kernel/Adam_1/Initializer/zeros*
dtype0
з
(conv2d/kernel/Adam_1/Read/ReadVariableOpReadVariableOpconv2d/kernel/Adam_1* 
_class
loc:@conv2d/kernel*&
_output_shapes
:*
dtype0
│
6conv2d_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_1/kernel*
_output_shapes
:*
dtype0*%
valueB"         
   
Х
,conv2d_1/kernel/Adam/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
√
&conv2d_1/kernel/Adam/Initializer/zerosFill6conv2d_1/kernel/Adam/Initializer/zeros/shape_as_tensor,conv2d_1/kernel/Adam/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
*

index_type0
┴
conv2d_1/kernel/AdamVarHandleOp*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
	container *
dtype0*
shape:
*%
shared_nameconv2d_1/kernel/Adam
Э
5conv2d_1/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/kernel/Adam*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
z
conv2d_1/kernel/Adam/AssignAssignVariableOpconv2d_1/kernel/Adam&conv2d_1/kernel/Adam/Initializer/zeros*
dtype0
й
(conv2d_1/kernel/Adam/Read/ReadVariableOpReadVariableOpconv2d_1/kernel/Adam*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
*
dtype0
╡
8conv2d_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_1/kernel*
_output_shapes
:*
dtype0*%
valueB"         
   
Ч
.conv2d_1/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Б
(conv2d_1/kernel/Adam_1/Initializer/zerosFill8conv2d_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor.conv2d_1/kernel/Adam_1/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
*

index_type0
┼
conv2d_1/kernel/Adam_1VarHandleOp*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
	container *
dtype0*
shape:
*'
shared_nameconv2d_1/kernel/Adam_1
б
7conv2d_1/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/kernel/Adam_1*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
А
conv2d_1/kernel/Adam_1/AssignAssignVariableOpconv2d_1/kernel/Adam_1(conv2d_1/kernel/Adam_1/Initializer/zeros*
dtype0
н
*conv2d_1/kernel/Adam_1/Read/ReadVariableOpReadVariableOpconv2d_1/kernel/Adam_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
*
dtype0
│
6conv2d_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_2/kernel*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
Х
,conv2d_2/kernel/Adam/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
№
&conv2d_2/kernel/Adam/Initializer/zerosFill6conv2d_2/kernel/Adam/Initializer/zeros/shape_as_tensor,conv2d_2/kernel/Adam/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_2/kernel*'
_output_shapes
:1
А*

index_type0
┬
conv2d_2/kernel/AdamVarHandleOp*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: *
	container *
dtype0*
shape:1
А*%
shared_nameconv2d_2/kernel/Adam
Э
5conv2d_2/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/kernel/Adam*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
z
conv2d_2/kernel/Adam/AssignAssignVariableOpconv2d_2/kernel/Adam&conv2d_2/kernel/Adam/Initializer/zeros*
dtype0
к
(conv2d_2/kernel/Adam/Read/ReadVariableOpReadVariableOpconv2d_2/kernel/Adam*"
_class
loc:@conv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
╡
8conv2d_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_2/kernel*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
Ч
.conv2d_2/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
В
(conv2d_2/kernel/Adam_1/Initializer/zerosFill8conv2d_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor.conv2d_2/kernel/Adam_1/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_2/kernel*'
_output_shapes
:1
А*

index_type0
╞
conv2d_2/kernel/Adam_1VarHandleOp*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: *
	container *
dtype0*
shape:1
А*'
shared_nameconv2d_2/kernel/Adam_1
б
7conv2d_2/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/kernel/Adam_1*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
А
conv2d_2/kernel/Adam_1/AssignAssignVariableOpconv2d_2/kernel/Adam_1(conv2d_2/kernel/Adam_1/Initializer/zeros*
dtype0
о
*conv2d_2/kernel/Adam_1/Read/ReadVariableOpReadVariableOpconv2d_2/kernel/Adam_1*"
_class
loc:@conv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
│
6conv2d_3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_3/kernel*
_output_shapes
:*
dtype0*%
valueB"А         А   
Х
,conv2d_3/kernel/Adam/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
¤
&conv2d_3/kernel/Adam/Initializer/zerosFill6conv2d_3/kernel/Adam/Initializer/zeros/shape_as_tensor,conv2d_3/kernel/Adam/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_3/kernel*(
_output_shapes
:АА*

index_type0
├
conv2d_3/kernel/AdamVarHandleOp*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: *
	container *
dtype0*
shape:АА*%
shared_nameconv2d_3/kernel/Adam
Э
5conv2d_3/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/kernel/Adam*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
z
conv2d_3/kernel/Adam/AssignAssignVariableOpconv2d_3/kernel/Adam&conv2d_3/kernel/Adam/Initializer/zeros*
dtype0
л
(conv2d_3/kernel/Adam/Read/ReadVariableOpReadVariableOpconv2d_3/kernel/Adam*"
_class
loc:@conv2d_3/kernel*(
_output_shapes
:АА*
dtype0
╡
8conv2d_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_3/kernel*
_output_shapes
:*
dtype0*%
valueB"А         А   
Ч
.conv2d_3/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Г
(conv2d_3/kernel/Adam_1/Initializer/zerosFill8conv2d_3/kernel/Adam_1/Initializer/zeros/shape_as_tensor.conv2d_3/kernel/Adam_1/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_3/kernel*(
_output_shapes
:АА*

index_type0
╟
conv2d_3/kernel/Adam_1VarHandleOp*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: *
	container *
dtype0*
shape:АА*'
shared_nameconv2d_3/kernel/Adam_1
б
7conv2d_3/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/kernel/Adam_1*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
А
conv2d_3/kernel/Adam_1/AssignAssignVariableOpconv2d_3/kernel/Adam_1(conv2d_3/kernel/Adam_1/Initializer/zeros*
dtype0
п
*conv2d_3/kernel/Adam_1/Read/ReadVariableOpReadVariableOpconv2d_3/kernel/Adam_1*"
_class
loc:@conv2d_3/kernel*(
_output_shapes
:АА*
dtype0
│
6conv2d_4/kernel/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_4/kernel*
_output_shapes
:*
dtype0*%
valueB"А         
   
Х
,conv2d_4/kernel/Adam/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
№
&conv2d_4/kernel/Adam/Initializer/zerosFill6conv2d_4/kernel/Adam/Initializer/zeros/shape_as_tensor,conv2d_4/kernel/Adam/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_4/kernel*'
_output_shapes
:А
*

index_type0
┬
conv2d_4/kernel/AdamVarHandleOp*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: *
	container *
dtype0*
shape:А
*%
shared_nameconv2d_4/kernel/Adam
Э
5conv2d_4/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/kernel/Adam*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: 
z
conv2d_4/kernel/Adam/AssignAssignVariableOpconv2d_4/kernel/Adam&conv2d_4/kernel/Adam/Initializer/zeros*
dtype0
к
(conv2d_4/kernel/Adam/Read/ReadVariableOpReadVariableOpconv2d_4/kernel/Adam*"
_class
loc:@conv2d_4/kernel*'
_output_shapes
:А
*
dtype0
╡
8conv2d_4/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_4/kernel*
_output_shapes
:*
dtype0*%
valueB"А         
   
Ч
.conv2d_4/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
В
(conv2d_4/kernel/Adam_1/Initializer/zerosFill8conv2d_4/kernel/Adam_1/Initializer/zeros/shape_as_tensor.conv2d_4/kernel/Adam_1/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_4/kernel*'
_output_shapes
:А
*

index_type0
╞
conv2d_4/kernel/Adam_1VarHandleOp*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: *
	container *
dtype0*
shape:А
*'
shared_nameconv2d_4/kernel/Adam_1
б
7conv2d_4/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/kernel/Adam_1*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: 
А
conv2d_4/kernel/Adam_1/AssignAssignVariableOpconv2d_4/kernel/Adam_1(conv2d_4/kernel/Adam_1/Initializer/zeros*
dtype0
о
*conv2d_4/kernel/Adam_1/Read/ReadVariableOpReadVariableOpconv2d_4/kernel/Adam_1*"
_class
loc:@conv2d_4/kernel*'
_output_shapes
:А
*
dtype0
O

Adam/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
O

Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *w╛?
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *w╠+2
z
6Adam/update_intercept/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
|
8Adam/update_intercept/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ц
'Adam/update_intercept/ResourceApplyAdamResourceApplyAdam	interceptintercept/Adamintercept/Adam_16Adam/update_intercept/ResourceApplyAdam/ReadVariableOp8Adam/update_intercept/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_20*
T0*
_class
loc:@intercept*
use_locking( *
use_nesterov( 
v
2Adam/update_slope/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
x
4Adam/update_slope/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
╩
#Adam/update_slope/ResourceApplyAdamResourceApplyAdamslope
slope/Adamslope/Adam_12Adam/update_slope/ResourceApplyAdam/ReadVariableOp4Adam/update_slope/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_22*
T0*
_class

loc:@slope*
use_locking( *
use_nesterov( 
|
8Adam/update_intercept_1/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
~
:Adam/update_intercept_1/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Ї
)Adam/update_intercept_1/ResourceApplyAdamResourceApplyAdamintercept_1intercept_1/Adamintercept_1/Adam_18Adam/update_intercept_1/ResourceApplyAdam/ReadVariableOp:Adam/update_intercept_1/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_18*
T0*
_class
loc:@intercept_1*
use_locking( *
use_nesterov( 
x
4Adam/update_slope_1/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
z
6Adam/update_slope_1/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
╪
%Adam/update_slope_1/ResourceApplyAdamResourceApplyAdamslope_1slope_1/Adamslope_1/Adam_14Adam/update_slope_1/ResourceApplyAdam/ReadVariableOp6Adam/update_slope_1/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_19*
T0*
_class
loc:@slope_1*
use_locking( *
use_nesterov( 
|
8Adam/update_intercept_2/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
~
:Adam/update_intercept_2/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Ї
)Adam/update_intercept_2/ResourceApplyAdamResourceApplyAdamintercept_2intercept_2/Adamintercept_2/Adam_18Adam/update_intercept_2/ResourceApplyAdam/ReadVariableOp:Adam/update_intercept_2/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_13*
T0*
_class
loc:@intercept_2*
use_locking( *
use_nesterov( 
x
4Adam/update_slope_2/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
z
6Adam/update_slope_2/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
╪
%Adam/update_slope_2/ResourceApplyAdamResourceApplyAdamslope_2slope_2/Adamslope_2/Adam_14Adam/update_slope_2/ResourceApplyAdam/ReadVariableOp6Adam/update_slope_2/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_14*
T0*
_class
loc:@slope_2*
use_locking( *
use_nesterov( 
|
8Adam/update_intercept_3/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
~
:Adam/update_intercept_3/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
є
)Adam/update_intercept_3/ResourceApplyAdamResourceApplyAdamintercept_3intercept_3/Adamintercept_3/Adam_18Adam/update_intercept_3/ResourceApplyAdam/ReadVariableOp:Adam/update_intercept_3/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
T0*
_class
loc:@intercept_3*
use_locking( *
use_nesterov( 
x
4Adam/update_slope_3/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
z
6Adam/update_slope_3/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
╫
%Adam/update_slope_3/ResourceApplyAdamResourceApplyAdamslope_3slope_3/Adamslope_3/Adam_14Adam/update_slope_3/ResourceApplyAdam/ReadVariableOp6Adam/update_slope_3/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
T0*
_class
loc:@slope_3*
use_locking( *
use_nesterov( 
|
8Adam/update_intercept_4/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
~
:Adam/update_intercept_4/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
є
)Adam/update_intercept_4/ResourceApplyAdamResourceApplyAdamintercept_4intercept_4/Adamintercept_4/Adam_18Adam/update_intercept_4/ResourceApplyAdam/ReadVariableOp:Adam/update_intercept_4/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_6*
T0*
_class
loc:@intercept_4*
use_locking( *
use_nesterov( 
x
4Adam/update_slope_4/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
z
6Adam/update_slope_4/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
╫
%Adam/update_slope_4/ResourceApplyAdamResourceApplyAdamslope_4slope_4/Adamslope_4/Adam_14Adam/update_slope_4/ResourceApplyAdam/ReadVariableOp6Adam/update_slope_4/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_9*
T0*
_class
loc:@slope_4*
use_locking( *
use_nesterov( 
|
8Adam/update_intercept_5/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
~
:Adam/update_intercept_5/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
є
)Adam/update_intercept_5/ResourceApplyAdamResourceApplyAdamintercept_5intercept_5/Adamintercept_5/Adam_18Adam/update_intercept_5/ResourceApplyAdam/ReadVariableOp:Adam/update_intercept_5/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_7*
T0*
_class
loc:@intercept_5*
use_locking( *
use_nesterov( 
x
4Adam/update_slope_5/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
z
6Adam/update_slope_5/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
╪
%Adam/update_slope_5/ResourceApplyAdamResourceApplyAdamslope_5slope_5/Adamslope_5/Adam_14Adam/update_slope_5/ResourceApplyAdam/ReadVariableOp6Adam/update_slope_5/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_10*
T0*
_class
loc:@slope_5*
use_locking( *
use_nesterov( 
~
:Adam/update_conv2d/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
А
<Adam/update_conv2d/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
В
+Adam/update_conv2d/kernel/ResourceApplyAdamResourceApplyAdamconv2d/kernelconv2d/kernel/Adamconv2d/kernel/Adam_1:Adam/update_conv2d/kernel/ResourceApplyAdam/ReadVariableOp<Adam/update_conv2d/kernel/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_23*
T0* 
_class
loc:@conv2d/kernel*
use_locking( *
use_nesterov( 
А
<Adam/update_conv2d_1/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
В
>Adam/update_conv2d_1/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Р
-Adam/update_conv2d_1/kernel/ResourceApplyAdamResourceApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1<Adam/update_conv2d_1/kernel/ResourceApplyAdam/ReadVariableOp>Adam/update_conv2d_1/kernel/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_21*
T0*"
_class
loc:@conv2d_1/kernel*
use_locking( *
use_nesterov( 
А
<Adam/update_conv2d_2/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
В
>Adam/update_conv2d_2/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Р
-Adam/update_conv2d_2/kernel/ResourceApplyAdamResourceApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1<Adam/update_conv2d_2/kernel/ResourceApplyAdam/ReadVariableOp>Adam/update_conv2d_2/kernel/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_16*
T0*"
_class
loc:@conv2d_2/kernel*
use_locking( *
use_nesterov( 
А
<Adam/update_conv2d_3/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
В
>Adam/update_conv2d_3/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Р
-Adam/update_conv2d_3/kernel/ResourceApplyAdamResourceApplyAdamconv2d_3/kernelconv2d_3/kernel/Adamconv2d_3/kernel/Adam_1<Adam/update_conv2d_3/kernel/ResourceApplyAdam/ReadVariableOp>Adam/update_conv2d_3/kernel/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_11*
T0*"
_class
loc:@conv2d_3/kernel*
use_locking( *
use_nesterov( 
А
<Adam/update_conv2d_4/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
В
>Adam/update_conv2d_4/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Р
-Adam/update_conv2d_4/kernel/ResourceApplyAdamResourceApplyAdamconv2d_4/kernelconv2d_4/kernel/Adamconv2d_4/kernel/Adam_1<Adam/update_conv2d_4/kernel/ResourceApplyAdam/ReadVariableOp>Adam/update_conv2d_4/kernel/ResourceApplyAdam/ReadVariableOp_1lr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_12*
T0*"
_class
loc:@conv2d_4/kernel*
use_locking( *
use_nesterov( 
╣
Adam/ReadVariableOpReadVariableOpbeta1_power,^Adam/update_conv2d/kernel/ResourceApplyAdam.^Adam/update_conv2d_1/kernel/ResourceApplyAdam.^Adam/update_conv2d_2/kernel/ResourceApplyAdam.^Adam/update_conv2d_3/kernel/ResourceApplyAdam.^Adam/update_conv2d_4/kernel/ResourceApplyAdam(^Adam/update_intercept/ResourceApplyAdam*^Adam/update_intercept_1/ResourceApplyAdam*^Adam/update_intercept_2/ResourceApplyAdam*^Adam/update_intercept_3/ResourceApplyAdam*^Adam/update_intercept_4/ResourceApplyAdam*^Adam/update_intercept_5/ResourceApplyAdam$^Adam/update_slope/ResourceApplyAdam&^Adam/update_slope_1/ResourceApplyAdam&^Adam/update_slope_2/ResourceApplyAdam&^Adam/update_slope_3/ResourceApplyAdam&^Adam/update_slope_4/ResourceApplyAdam&^Adam/update_slope_5/ResourceApplyAdam*
_output_shapes
: *
dtype0
s
Adam/mulMulAdam/ReadVariableOp
Adam/beta1*
T0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
o
Adam/AssignVariableOpAssignVariableOpbeta1_powerAdam/mul* 
_class
loc:@conv2d/kernel*
dtype0
ї
Adam/ReadVariableOp_1ReadVariableOpbeta1_power^Adam/AssignVariableOp,^Adam/update_conv2d/kernel/ResourceApplyAdam.^Adam/update_conv2d_1/kernel/ResourceApplyAdam.^Adam/update_conv2d_2/kernel/ResourceApplyAdam.^Adam/update_conv2d_3/kernel/ResourceApplyAdam.^Adam/update_conv2d_4/kernel/ResourceApplyAdam(^Adam/update_intercept/ResourceApplyAdam*^Adam/update_intercept_1/ResourceApplyAdam*^Adam/update_intercept_2/ResourceApplyAdam*^Adam/update_intercept_3/ResourceApplyAdam*^Adam/update_intercept_4/ResourceApplyAdam*^Adam/update_intercept_5/ResourceApplyAdam$^Adam/update_slope/ResourceApplyAdam&^Adam/update_slope_1/ResourceApplyAdam&^Adam/update_slope_2/ResourceApplyAdam&^Adam/update_slope_3/ResourceApplyAdam&^Adam/update_slope_4/ResourceApplyAdam&^Adam/update_slope_5/ResourceApplyAdam* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
dtype0
╗
Adam/ReadVariableOp_2ReadVariableOpbeta2_power,^Adam/update_conv2d/kernel/ResourceApplyAdam.^Adam/update_conv2d_1/kernel/ResourceApplyAdam.^Adam/update_conv2d_2/kernel/ResourceApplyAdam.^Adam/update_conv2d_3/kernel/ResourceApplyAdam.^Adam/update_conv2d_4/kernel/ResourceApplyAdam(^Adam/update_intercept/ResourceApplyAdam*^Adam/update_intercept_1/ResourceApplyAdam*^Adam/update_intercept_2/ResourceApplyAdam*^Adam/update_intercept_3/ResourceApplyAdam*^Adam/update_intercept_4/ResourceApplyAdam*^Adam/update_intercept_5/ResourceApplyAdam$^Adam/update_slope/ResourceApplyAdam&^Adam/update_slope_1/ResourceApplyAdam&^Adam/update_slope_2/ResourceApplyAdam&^Adam/update_slope_3/ResourceApplyAdam&^Adam/update_slope_4/ResourceApplyAdam&^Adam/update_slope_5/ResourceApplyAdam*
_output_shapes
: *
dtype0
w

Adam/mul_1MulAdam/ReadVariableOp_2
Adam/beta2*
T0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
s
Adam/AssignVariableOp_1AssignVariableOpbeta2_power
Adam/mul_1* 
_class
loc:@conv2d/kernel*
dtype0
ў
Adam/ReadVariableOp_3ReadVariableOpbeta2_power^Adam/AssignVariableOp_1,^Adam/update_conv2d/kernel/ResourceApplyAdam.^Adam/update_conv2d_1/kernel/ResourceApplyAdam.^Adam/update_conv2d_2/kernel/ResourceApplyAdam.^Adam/update_conv2d_3/kernel/ResourceApplyAdam.^Adam/update_conv2d_4/kernel/ResourceApplyAdam(^Adam/update_intercept/ResourceApplyAdam*^Adam/update_intercept_1/ResourceApplyAdam*^Adam/update_intercept_2/ResourceApplyAdam*^Adam/update_intercept_3/ResourceApplyAdam*^Adam/update_intercept_4/ResourceApplyAdam*^Adam/update_intercept_5/ResourceApplyAdam$^Adam/update_slope/ResourceApplyAdam&^Adam/update_slope_1/ResourceApplyAdam&^Adam/update_slope_2/ResourceApplyAdam&^Adam/update_slope_3/ResourceApplyAdam&^Adam/update_slope_4/ResourceApplyAdam&^Adam/update_slope_5/ResourceApplyAdam* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
dtype0
а
AdamNoOp^Adam/AssignVariableOp^Adam/AssignVariableOp_1,^Adam/update_conv2d/kernel/ResourceApplyAdam.^Adam/update_conv2d_1/kernel/ResourceApplyAdam.^Adam/update_conv2d_2/kernel/ResourceApplyAdam.^Adam/update_conv2d_3/kernel/ResourceApplyAdam.^Adam/update_conv2d_4/kernel/ResourceApplyAdam(^Adam/update_intercept/ResourceApplyAdam*^Adam/update_intercept_1/ResourceApplyAdam*^Adam/update_intercept_2/ResourceApplyAdam*^Adam/update_intercept_3/ResourceApplyAdam*^Adam/update_intercept_4/ResourceApplyAdam*^Adam/update_intercept_5/ResourceApplyAdam$^Adam/update_slope/ResourceApplyAdam&^Adam/update_slope_1/ResourceApplyAdam&^Adam/update_slope_2/ResourceApplyAdam&^Adam/update_slope_3/ResourceApplyAdam&^Adam/update_slope_4/ResourceApplyAdam&^Adam/update_slope_5/ResourceApplyAdam
R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
~
ArgMaxArgMaxPlaceholder_1ArgMax/dimension*
T0*

Tidx0*#
_output_shapes
:         *
output_type0	
T
ArgMax_1/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
И
ArgMax_1ArgMaxflatten_1/Reshape_1ArgMax_1/dimension*
T0*

Tidx0*#
_output_shapes
:         *
output_type0	
n
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:         *
incompatible_shape_error(
`
CastCastEqual*

DstT0*

SrcT0
*
Truncate( *#
_output_shapes
:         
R
Const_35Const*
_output_shapes
:*
dtype0*
valueB: 
\
Mean_2MeanCastConst_35*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
T
ArgMax_2/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
В
ArgMax_2ArgMaxPlaceholder_1ArgMax_2/dimension*
T0*

Tidx0*#
_output_shapes
:         *
output_type0	
T
ArgMax_3/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
Ж
ArgMax_3ArgMaxflatten_1/ReshapeArgMax_3/dimension*
T0*

Tidx0*#
_output_shapes
:         *
output_type0	
r
Equal_1EqualArgMax_2ArgMax_3*
T0	*#
_output_shapes
:         *
incompatible_shape_error(
d
Cast_1CastEqual_1*

DstT0*

SrcT0
*
Truncate( *#
_output_shapes
:         
R
Const_36Const*
_output_shapes
:*
dtype0*
valueB: 
^
Mean_3MeanCast_1Const_36*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
L
Abs_18Absconv2d/mul_2*
T0*&
_output_shapes
:
M
add_55/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
R
add_55AddV2Abs_18add_55/y*
T0*&
_output_shapes
:
F
Log_36Logadd_55*
T0*&
_output_shapes
:
M
Const_37Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_37LogConst_37*
T0*
_output_shapes
: 
V

truediv_18RealDivLog_36Log_37*
T0*&
_output_shapes
:
O
ReadVariableOp_24ReadVariableOpslope*
_output_shapes
: *
dtype0
]
mul_15MulReadVariableOp_24
truediv_18*
T0*&
_output_shapes
:
S
ReadVariableOp_25ReadVariableOp	intercept*
_output_shapes
: *
dtype0
[
add_56AddV2ReadVariableOp_25mul_15*
T0*&
_output_shapes
:
z
differentiable_round_12Roundadd_56*
T0*
_gradient_op_type
Identity*&
_output_shapes
:
a
Const_38Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Min_12Mindifferentiable_round_12Const_38*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_15/packedPackMin_12*
N*
T0*
_output_shapes
:*

axis 
I
Rank_15Const*
_output_shapes
: *
dtype0*
value	B :
P
range_15/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_15/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_15Rangerange_15/startRank_15range_15/delta*

Tidx0*
_output_shapes
:
V
Min_13/inputPackMin_12*
N*
T0*
_output_shapes
:*

axis 
c
Min_13MinMin_13/inputrange_15*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
L
Abs_19Absconv2d/mul_2*
T0*&
_output_shapes
:
M
add_57/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
R
add_57AddV2Abs_19add_57/y*
T0*&
_output_shapes
:
F
Log_38Logadd_57*
T0*&
_output_shapes
:
M
Const_39Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_39LogConst_39*
T0*
_output_shapes
: 
V

truediv_19RealDivLog_38Log_39*
T0*&
_output_shapes
:
O
ReadVariableOp_26ReadVariableOpslope*
_output_shapes
: *
dtype0
]
mul_16MulReadVariableOp_26
truediv_19*
T0*&
_output_shapes
:
S
ReadVariableOp_27ReadVariableOp	intercept*
_output_shapes
: *
dtype0
[
add_58AddV2ReadVariableOp_27mul_16*
T0*&
_output_shapes
:
z
differentiable_round_13Roundadd_58*
T0*
_gradient_op_type
Identity*&
_output_shapes
:
a
Const_40Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Max_12Maxdifferentiable_round_13Const_40*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_16/packedPackMax_12*
N*
T0*
_output_shapes
:*

axis 
I
Rank_16Const*
_output_shapes
: *
dtype0*
value	B :
P
range_16/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_16/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_16Rangerange_16/startRank_16range_16/delta*

Tidx0*
_output_shapes
:
V
Max_13/inputPackMax_12*
N*
T0*
_output_shapes
:*

axis 
c
Max_13MaxMax_13/inputrange_16*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
=
sub_6SubMax_13Min_13*
T0*
_output_shapes
: 
M
add_59/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
A
add_59AddV2sub_6add_59/y*
T0*
_output_shapes
: 
6
Abs_20Absadd_59*
T0*
_output_shapes
: 
M
add_60/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
B
add_60AddV2Abs_20add_60/y*
T0*
_output_shapes
: 
6
Log_40Logadd_60*
T0*
_output_shapes
: 
M
Const_41Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_41LogConst_41*
T0*
_output_shapes
: 
F

truediv_20RealDivLog_40Log_41*
T0*
_output_shapes
: 
k
differentiable_ceil_6Ceil
truediv_20*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
M
add_61/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Q
add_61AddV2add_61/xdifferentiable_ceil_6*
T0*
_output_shapes
: 
N
Abs_21Absconv2d_1/mul_2*
T0*&
_output_shapes
:

M
add_62/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
R
add_62AddV2Abs_21add_62/y*
T0*&
_output_shapes
:

F
Log_42Logadd_62*
T0*&
_output_shapes
:

M
Const_42Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_43LogConst_42*
T0*
_output_shapes
: 
V

truediv_21RealDivLog_42Log_43*
T0*&
_output_shapes
:

Q
ReadVariableOp_28ReadVariableOpslope_1*
_output_shapes
: *
dtype0
]
mul_17MulReadVariableOp_28
truediv_21*
T0*&
_output_shapes
:

U
ReadVariableOp_29ReadVariableOpintercept_1*
_output_shapes
: *
dtype0
[
add_63AddV2ReadVariableOp_29mul_17*
T0*&
_output_shapes
:

z
differentiable_round_14Roundadd_63*
T0*
_gradient_op_type
Identity*&
_output_shapes
:

a
Const_43Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Min_14Mindifferentiable_round_14Const_43*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_17/packedPackMin_14*
N*
T0*
_output_shapes
:*

axis 
I
Rank_17Const*
_output_shapes
: *
dtype0*
value	B :
P
range_17/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_17/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_17Rangerange_17/startRank_17range_17/delta*

Tidx0*
_output_shapes
:
V
Min_15/inputPackMin_14*
N*
T0*
_output_shapes
:*

axis 
c
Min_15MinMin_15/inputrange_17*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
N
Abs_22Absconv2d_1/mul_2*
T0*&
_output_shapes
:

M
add_64/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
R
add_64AddV2Abs_22add_64/y*
T0*&
_output_shapes
:

F
Log_44Logadd_64*
T0*&
_output_shapes
:

M
Const_44Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_45LogConst_44*
T0*
_output_shapes
: 
V

truediv_22RealDivLog_44Log_45*
T0*&
_output_shapes
:

Q
ReadVariableOp_30ReadVariableOpslope_1*
_output_shapes
: *
dtype0
]
mul_18MulReadVariableOp_30
truediv_22*
T0*&
_output_shapes
:

U
ReadVariableOp_31ReadVariableOpintercept_1*
_output_shapes
: *
dtype0
[
add_65AddV2ReadVariableOp_31mul_18*
T0*&
_output_shapes
:

z
differentiable_round_15Roundadd_65*
T0*
_gradient_op_type
Identity*&
_output_shapes
:

a
Const_45Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Max_14Maxdifferentiable_round_15Const_45*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_18/packedPackMax_14*
N*
T0*
_output_shapes
:*

axis 
I
Rank_18Const*
_output_shapes
: *
dtype0*
value	B :
P
range_18/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_18/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_18Rangerange_18/startRank_18range_18/delta*

Tidx0*
_output_shapes
:
V
Max_15/inputPackMax_14*
N*
T0*
_output_shapes
:*

axis 
c
Max_15MaxMax_15/inputrange_18*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
=
sub_7SubMax_15Min_15*
T0*
_output_shapes
: 
M
add_66/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
A
add_66AddV2sub_7add_66/y*
T0*
_output_shapes
: 
6
Abs_23Absadd_66*
T0*
_output_shapes
: 
M
add_67/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
B
add_67AddV2Abs_23add_67/y*
T0*
_output_shapes
: 
6
Log_46Logadd_67*
T0*
_output_shapes
: 
M
Const_46Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_47LogConst_46*
T0*
_output_shapes
: 
F

truediv_23RealDivLog_46Log_47*
T0*
_output_shapes
: 
k
differentiable_ceil_7Ceil
truediv_23*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
M
add_68/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Q
add_68AddV2add_68/xdifferentiable_ceil_7*
T0*
_output_shapes
: 
O
Abs_24Absconv2d_2/mul_2*
T0*'
_output_shapes
:1
А
M
add_69/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
S
add_69AddV2Abs_24add_69/y*
T0*'
_output_shapes
:1
А
G
Log_48Logadd_69*
T0*'
_output_shapes
:1
А
M
Const_47Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_49LogConst_47*
T0*
_output_shapes
: 
W

truediv_24RealDivLog_48Log_49*
T0*'
_output_shapes
:1
А
Q
ReadVariableOp_32ReadVariableOpslope_2*
_output_shapes
: *
dtype0
^
mul_19MulReadVariableOp_32
truediv_24*
T0*'
_output_shapes
:1
А
U
ReadVariableOp_33ReadVariableOpintercept_2*
_output_shapes
: *
dtype0
\
add_70AddV2ReadVariableOp_33mul_19*
T0*'
_output_shapes
:1
А
{
differentiable_round_16Roundadd_70*
T0*
_gradient_op_type
Identity*'
_output_shapes
:1
А
a
Const_48Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Min_16Mindifferentiable_round_16Const_48*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_19/packedPackMin_16*
N*
T0*
_output_shapes
:*

axis 
I
Rank_19Const*
_output_shapes
: *
dtype0*
value	B :
P
range_19/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_19/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_19Rangerange_19/startRank_19range_19/delta*

Tidx0*
_output_shapes
:
V
Min_17/inputPackMin_16*
N*
T0*
_output_shapes
:*

axis 
c
Min_17MinMin_17/inputrange_19*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
O
Abs_25Absconv2d_2/mul_2*
T0*'
_output_shapes
:1
А
M
add_71/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
S
add_71AddV2Abs_25add_71/y*
T0*'
_output_shapes
:1
А
G
Log_50Logadd_71*
T0*'
_output_shapes
:1
А
M
Const_49Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_51LogConst_49*
T0*
_output_shapes
: 
W

truediv_25RealDivLog_50Log_51*
T0*'
_output_shapes
:1
А
Q
ReadVariableOp_34ReadVariableOpslope_2*
_output_shapes
: *
dtype0
^
mul_20MulReadVariableOp_34
truediv_25*
T0*'
_output_shapes
:1
А
U
ReadVariableOp_35ReadVariableOpintercept_2*
_output_shapes
: *
dtype0
\
add_72AddV2ReadVariableOp_35mul_20*
T0*'
_output_shapes
:1
А
{
differentiable_round_17Roundadd_72*
T0*
_gradient_op_type
Identity*'
_output_shapes
:1
А
a
Const_50Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Max_16Maxdifferentiable_round_17Const_50*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_20/packedPackMax_16*
N*
T0*
_output_shapes
:*

axis 
I
Rank_20Const*
_output_shapes
: *
dtype0*
value	B :
P
range_20/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_20/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_20Rangerange_20/startRank_20range_20/delta*

Tidx0*
_output_shapes
:
V
Max_17/inputPackMax_16*
N*
T0*
_output_shapes
:*

axis 
c
Max_17MaxMax_17/inputrange_20*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
=
sub_8SubMax_17Min_17*
T0*
_output_shapes
: 
M
add_73/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
A
add_73AddV2sub_8add_73/y*
T0*
_output_shapes
: 
6
Abs_26Absadd_73*
T0*
_output_shapes
: 
M
add_74/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
B
add_74AddV2Abs_26add_74/y*
T0*
_output_shapes
: 
6
Log_52Logadd_74*
T0*
_output_shapes
: 
M
Const_51Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_53LogConst_51*
T0*
_output_shapes
: 
F

truediv_26RealDivLog_52Log_53*
T0*
_output_shapes
: 
k
differentiable_ceil_8Ceil
truediv_26*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
M
add_75/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Q
add_75AddV2add_75/xdifferentiable_ceil_8*
T0*
_output_shapes
: 
Y
Abs_27Absconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
M
add_76/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
\
add_76AddV2Abs_27add_76/y*
T0*0
_output_shapes
:         А
P
Log_54Logadd_76*
T0*0
_output_shapes
:         А
M
Const_52Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_55LogConst_52*
T0*
_output_shapes
: 
`

truediv_27RealDivLog_54Log_55*
T0*0
_output_shapes
:         А
Q
ReadVariableOp_36ReadVariableOpslope_3*
_output_shapes
: *
dtype0
g
mul_21MulReadVariableOp_36
truediv_27*
T0*0
_output_shapes
:         А
U
ReadVariableOp_37ReadVariableOpintercept_3*
_output_shapes
: *
dtype0
e
add_77AddV2ReadVariableOp_37mul_21*
T0*0
_output_shapes
:         А
Д
differentiable_round_18Roundadd_77*
T0*
_gradient_op_type
Identity*0
_output_shapes
:         А
a
Const_53Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Min_18Mindifferentiable_round_18Const_53*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_21/packedPackMin_18*
N*
T0*
_output_shapes
:*

axis 
I
Rank_21Const*
_output_shapes
: *
dtype0*
value	B :
P
range_21/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_21/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_21Rangerange_21/startRank_21range_21/delta*

Tidx0*
_output_shapes
:
V
Min_19/inputPackMin_18*
N*
T0*
_output_shapes
:*

axis 
c
Min_19MinMin_19/inputrange_21*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Y
Abs_28Absconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
M
add_78/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
\
add_78AddV2Abs_28add_78/y*
T0*0
_output_shapes
:         А
P
Log_56Logadd_78*
T0*0
_output_shapes
:         А
M
Const_54Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_57LogConst_54*
T0*
_output_shapes
: 
`

truediv_28RealDivLog_56Log_57*
T0*0
_output_shapes
:         А
Q
ReadVariableOp_38ReadVariableOpslope_3*
_output_shapes
: *
dtype0
g
mul_22MulReadVariableOp_38
truediv_28*
T0*0
_output_shapes
:         А
U
ReadVariableOp_39ReadVariableOpintercept_3*
_output_shapes
: *
dtype0
e
add_79AddV2ReadVariableOp_39mul_22*
T0*0
_output_shapes
:         А
Д
differentiable_round_19Roundadd_79*
T0*
_gradient_op_type
Identity*0
_output_shapes
:         А
a
Const_55Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Max_18Maxdifferentiable_round_19Const_55*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_22/packedPackMax_18*
N*
T0*
_output_shapes
:*

axis 
I
Rank_22Const*
_output_shapes
: *
dtype0*
value	B :
P
range_22/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_22/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_22Rangerange_22/startRank_22range_22/delta*

Tidx0*
_output_shapes
:
V
Max_19/inputPackMax_18*
N*
T0*
_output_shapes
:*

axis 
c
Max_19MaxMax_19/inputrange_22*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
=
sub_9SubMax_19Min_19*
T0*
_output_shapes
: 
M
add_80/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
A
add_80AddV2sub_9add_80/y*
T0*
_output_shapes
: 
6
Abs_29Absadd_80*
T0*
_output_shapes
: 
M
add_81/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
B
add_81AddV2Abs_29add_81/y*
T0*
_output_shapes
: 
6
Log_58Logadd_81*
T0*
_output_shapes
: 
M
Const_56Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_59LogConst_56*
T0*
_output_shapes
: 
F

truediv_29RealDivLog_58Log_59*
T0*
_output_shapes
: 
k
differentiable_ceil_9Ceil
truediv_29*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
M
add_82/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Q
add_82AddV2add_82/xdifferentiable_ceil_9*
T0*
_output_shapes
: 
P
Abs_30Absconv2d_3/mul_2*
T0*(
_output_shapes
:АА
M
add_83/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_83AddV2Abs_30add_83/y*
T0*(
_output_shapes
:АА
H
Log_60Logadd_83*
T0*(
_output_shapes
:АА
M
Const_57Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_61LogConst_57*
T0*
_output_shapes
: 
X

truediv_30RealDivLog_60Log_61*
T0*(
_output_shapes
:АА
Q
ReadVariableOp_40ReadVariableOpslope_4*
_output_shapes
: *
dtype0
_
mul_23MulReadVariableOp_40
truediv_30*
T0*(
_output_shapes
:АА
U
ReadVariableOp_41ReadVariableOpintercept_4*
_output_shapes
: *
dtype0
]
add_84AddV2ReadVariableOp_41mul_23*
T0*(
_output_shapes
:АА
|
differentiable_round_20Roundadd_84*
T0*
_gradient_op_type
Identity*(
_output_shapes
:АА
a
Const_58Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Min_20Mindifferentiable_round_20Const_58*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_23/packedPackMin_20*
N*
T0*
_output_shapes
:*

axis 
I
Rank_23Const*
_output_shapes
: *
dtype0*
value	B :
P
range_23/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_23/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_23Rangerange_23/startRank_23range_23/delta*

Tidx0*
_output_shapes
:
V
Min_21/inputPackMin_20*
N*
T0*
_output_shapes
:*

axis 
c
Min_21MinMin_21/inputrange_23*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
P
Abs_31Absconv2d_3/mul_2*
T0*(
_output_shapes
:АА
M
add_85/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_85AddV2Abs_31add_85/y*
T0*(
_output_shapes
:АА
H
Log_62Logadd_85*
T0*(
_output_shapes
:АА
M
Const_59Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_63LogConst_59*
T0*
_output_shapes
: 
X

truediv_31RealDivLog_62Log_63*
T0*(
_output_shapes
:АА
Q
ReadVariableOp_42ReadVariableOpslope_4*
_output_shapes
: *
dtype0
_
mul_24MulReadVariableOp_42
truediv_31*
T0*(
_output_shapes
:АА
U
ReadVariableOp_43ReadVariableOpintercept_4*
_output_shapes
: *
dtype0
]
add_86AddV2ReadVariableOp_43mul_24*
T0*(
_output_shapes
:АА
|
differentiable_round_21Roundadd_86*
T0*
_gradient_op_type
Identity*(
_output_shapes
:АА
a
Const_60Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Max_20Maxdifferentiable_round_21Const_60*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_24/packedPackMax_20*
N*
T0*
_output_shapes
:*

axis 
I
Rank_24Const*
_output_shapes
: *
dtype0*
value	B :
P
range_24/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_24/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_24Rangerange_24/startRank_24range_24/delta*

Tidx0*
_output_shapes
:
V
Max_21/inputPackMax_20*
N*
T0*
_output_shapes
:*

axis 
c
Max_21MaxMax_21/inputrange_24*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
>
sub_10SubMax_21Min_21*
T0*
_output_shapes
: 
M
add_87/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
B
add_87AddV2sub_10add_87/y*
T0*
_output_shapes
: 
6
Abs_32Absadd_87*
T0*
_output_shapes
: 
M
add_88/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
B
add_88AddV2Abs_32add_88/y*
T0*
_output_shapes
: 
6
Log_64Logadd_88*
T0*
_output_shapes
: 
M
Const_61Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_65LogConst_61*
T0*
_output_shapes
: 
F

truediv_32RealDivLog_64Log_65*
T0*
_output_shapes
: 
l
differentiable_ceil_10Ceil
truediv_32*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
M
add_89/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
R
add_89AddV2add_89/xdifferentiable_ceil_10*
T0*
_output_shapes
: 
O
Abs_33Absconv2d_4/mul_2*
T0*'
_output_shapes
:А

M
add_90/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
S
add_90AddV2Abs_33add_90/y*
T0*'
_output_shapes
:А

G
Log_66Logadd_90*
T0*'
_output_shapes
:А

M
Const_62Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_67LogConst_62*
T0*
_output_shapes
: 
W

truediv_33RealDivLog_66Log_67*
T0*'
_output_shapes
:А

Q
ReadVariableOp_44ReadVariableOpslope_5*
_output_shapes
: *
dtype0
^
mul_25MulReadVariableOp_44
truediv_33*
T0*'
_output_shapes
:А

U
ReadVariableOp_45ReadVariableOpintercept_5*
_output_shapes
: *
dtype0
\
add_91AddV2ReadVariableOp_45mul_25*
T0*'
_output_shapes
:А

{
differentiable_round_22Roundadd_91*
T0*
_gradient_op_type
Identity*'
_output_shapes
:А

a
Const_63Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Min_22Mindifferentiable_round_22Const_63*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_25/packedPackMin_22*
N*
T0*
_output_shapes
:*

axis 
I
Rank_25Const*
_output_shapes
: *
dtype0*
value	B :
P
range_25/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_25/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_25Rangerange_25/startRank_25range_25/delta*

Tidx0*
_output_shapes
:
V
Min_23/inputPackMin_22*
N*
T0*
_output_shapes
:*

axis 
c
Min_23MinMin_23/inputrange_25*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
O
Abs_34Absconv2d_4/mul_2*
T0*'
_output_shapes
:А

M
add_92/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
S
add_92AddV2Abs_34add_92/y*
T0*'
_output_shapes
:А

G
Log_68Logadd_92*
T0*'
_output_shapes
:А

M
Const_64Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_69LogConst_64*
T0*
_output_shapes
: 
W

truediv_34RealDivLog_68Log_69*
T0*'
_output_shapes
:А

Q
ReadVariableOp_46ReadVariableOpslope_5*
_output_shapes
: *
dtype0
^
mul_26MulReadVariableOp_46
truediv_34*
T0*'
_output_shapes
:А

U
ReadVariableOp_47ReadVariableOpintercept_5*
_output_shapes
: *
dtype0
\
add_93AddV2ReadVariableOp_47mul_26*
T0*'
_output_shapes
:А

{
differentiable_round_23Roundadd_93*
T0*
_gradient_op_type
Identity*'
_output_shapes
:А

a
Const_65Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Max_22Maxdifferentiable_round_23Const_65*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_26/packedPackMax_22*
N*
T0*
_output_shapes
:*

axis 
I
Rank_26Const*
_output_shapes
: *
dtype0*
value	B :
P
range_26/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_26/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_26Rangerange_26/startRank_26range_26/delta*

Tidx0*
_output_shapes
:
V
Max_23/inputPackMax_22*
N*
T0*
_output_shapes
:*

axis 
c
Max_23MaxMax_23/inputrange_26*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
>
sub_11SubMax_23Min_23*
T0*
_output_shapes
: 
M
add_94/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
B
add_94AddV2sub_11add_94/y*
T0*
_output_shapes
: 
6
Abs_35Absadd_94*
T0*
_output_shapes
: 
M
add_95/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
B
add_95AddV2Abs_35add_95/y*
T0*
_output_shapes
: 
6
Log_70Logadd_95*
T0*
_output_shapes
: 
M
Const_66Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_71LogConst_66*
T0*
_output_shapes
: 
F

truediv_35RealDivLog_70Log_71*
T0*
_output_shapes
: 
l
differentiable_ceil_11Ceil
truediv_35*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
M
add_96/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
R
add_96AddV2add_96/xdifferentiable_ceil_11*
T0*
_output_shapes
: 
[
IdentityIdentityPlaceholder*
T0*/
_output_shapes
:         
b
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
n
	Reshape_3ReshapePlaceholderReshape_3/shape*
T0*
Tshape0*#
_output_shapes
:         
b
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
k
	Reshape_4ReshapeIdentityReshape_4/shape*
T0*
Tshape0*#
_output_shapes
:         
Q
sub_12Sub	Reshape_3	Reshape_4*
T0*#
_output_shapes
:         
M
norm/mulMulsub_12sub_12*
T0*#
_output_shapes
:         
T

norm/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
g
norm/SumSumnorm/mul
norm/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
@
	norm/SqrtSqrtnorm/Sum*
T0*
_output_shapes
:
W
norm/SqueezeSqueeze	norm/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
]

Identity_1IdentityPlaceholder*
T0*/
_output_shapes
:         
b
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
n
	Reshape_5ReshapePlaceholderReshape_5/shape*
T0*
Tshape0*#
_output_shapes
:         
b
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
m
	Reshape_6Reshape
Identity_1Reshape_6/shape*
T0*
Tshape0*#
_output_shapes
:         
Q
sub_13Sub	Reshape_5	Reshape_6*
T0*#
_output_shapes
:         
O

norm_1/mulMulsub_13sub_13*
T0*#
_output_shapes
:         
V
norm_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
m

norm_1/SumSum
norm_1/mulnorm_1/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
D
norm_1/SqrtSqrt
norm_1/Sum*
T0*
_output_shapes
:
[
norm_1/SqueezeSqueezenorm_1/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
_

Identity_2Identityconv2d/Conv2D*
T0*/
_output_shapes
:         
b
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
p
	Reshape_7Reshapeconv2d/Conv2DReshape_7/shape*
T0*
Tshape0*#
_output_shapes
:         
b
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
m
	Reshape_8Reshape
Identity_2Reshape_8/shape*
T0*
Tshape0*#
_output_shapes
:         
Q
sub_14Sub	Reshape_7	Reshape_8*
T0*#
_output_shapes
:         
O

norm_2/mulMulsub_14sub_14*
T0*#
_output_shapes
:         
V
norm_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
m

norm_2/SumSum
norm_2/mulnorm_2/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
D
norm_2/SqrtSqrt
norm_2/Sum*
T0*
_output_shapes
:
[
norm_2/SqueezeSqueezenorm_2/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
_

Identity_3Identityconv2d/Conv2D*
T0*/
_output_shapes
:         
b
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
p
	Reshape_9Reshapeconv2d/Conv2DReshape_9/shape*
T0*
Tshape0*#
_output_shapes
:         
c
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
o

Reshape_10Reshape
Identity_3Reshape_10/shape*
T0*
Tshape0*#
_output_shapes
:         
R
sub_15Sub	Reshape_9
Reshape_10*
T0*#
_output_shapes
:         
O

norm_3/mulMulsub_15sub_15*
T0*#
_output_shapes
:         
V
norm_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
m

norm_3/SumSum
norm_3/mulnorm_3/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
D
norm_3/SqrtSqrt
norm_3/Sum*
T0*
_output_shapes
:
[
norm_3/SqueezeSqueezenorm_3/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
a

Identity_4Identityconv2d_1/Conv2D*
T0*/
_output_shapes
:         

c
Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
t

Reshape_11Reshapeconv2d_1/Conv2DReshape_11/shape*
T0*
Tshape0*#
_output_shapes
:         
c
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
o

Reshape_12Reshape
Identity_4Reshape_12/shape*
T0*
Tshape0*#
_output_shapes
:         
S
sub_16Sub
Reshape_11
Reshape_12*
T0*#
_output_shapes
:         
O

norm_4/mulMulsub_16sub_16*
T0*#
_output_shapes
:         
V
norm_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
m

norm_4/SumSum
norm_4/mulnorm_4/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
D
norm_4/SqrtSqrt
norm_4/Sum*
T0*
_output_shapes
:
[
norm_4/SqueezeSqueezenorm_4/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
a

Identity_5Identityconv2d_1/Conv2D*
T0*/
_output_shapes
:         

c
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
t

Reshape_13Reshapeconv2d_1/Conv2DReshape_13/shape*
T0*
Tshape0*#
_output_shapes
:         
c
Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
o

Reshape_14Reshape
Identity_5Reshape_14/shape*
T0*
Tshape0*#
_output_shapes
:         
S
sub_17Sub
Reshape_13
Reshape_14*
T0*#
_output_shapes
:         
O

norm_5/mulMulsub_17sub_17*
T0*#
_output_shapes
:         
V
norm_5/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
m

norm_5/SumSum
norm_5/mulnorm_5/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
D
norm_5/SqrtSqrt
norm_5/Sum*
T0*
_output_shapes
:
[
norm_5/SqueezeSqueezenorm_5/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
Y
Abs_36Absconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
M
add_97/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
\
add_97AddV2Abs_36add_97/y*
T0*0
_output_shapes
:         А
P
Log_72Logadd_97*
T0*0
_output_shapes
:         А
M
Const_67Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_73LogConst_67*
T0*
_output_shapes
: 
`

truediv_36RealDivLog_72Log_73*
T0*0
_output_shapes
:         А
Q
ReadVariableOp_48ReadVariableOpslope_3*
_output_shapes
: *
dtype0
g
mul_27MulReadVariableOp_48
truediv_36*
T0*0
_output_shapes
:         А
U
ReadVariableOp_49ReadVariableOpintercept_3*
_output_shapes
: *
dtype0
e
add_98AddV2ReadVariableOp_49mul_27*
T0*0
_output_shapes
:         А
Д
differentiable_round_24Roundadd_98*
T0*
_gradient_op_type
Identity*0
_output_shapes
:         А
S
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
x
GreaterEqualGreaterEqualconv2d_2/Conv2DGreaterEqual/y*
T0*0
_output_shapes
:         А
^
ones_like/ShapeShapeconv2d_2/Conv2D*
T0*
_output_shapes
:*
out_type0
T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
	ones_likeFillones_like/Shapeones_like/Const*
T0*0
_output_shapes
:         А*

index_type0
c

zeros_like	ZerosLikeconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
t
SelectV2SelectV2GreaterEqual	ones_like
zeros_like*
T0*0
_output_shapes
:         А
P
LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
o
	LessEqual	LessEqualconv2d_2/Conv2DLessEqual/y*
T0*0
_output_shapes
:         А
`
ones_like_1/ShapeShapeconv2d_2/Conv2D*
T0*
_output_shapes
:*
out_type0
V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ж
ones_like_1Fillones_like_1/Shapeones_like_1/Const*
T0*0
_output_shapes
:         А*

index_type0
M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
_
mul_28Mulmul_28/xones_like_1*
T0*0
_output_shapes
:         А
e
zeros_like_1	ZerosLikeconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
r

SelectV2_1SelectV2	LessEqualmul_28zeros_like_1*
T0*0
_output_shapes
:         А
^
Add_99Add
SelectV2_1SelectV2*
T0*0
_output_shapes
:         А
L
pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
i
pow_6Powpow_6/xdifferentiable_round_24*
T0*0
_output_shapes
:         А
W
mul_29MulAdd_99pow_6*
T0*0
_output_shapes
:         А
c
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
t

Reshape_15Reshapeconv2d_2/Conv2DReshape_15/shape*
T0*
Tshape0*#
_output_shapes
:         
c
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
k

Reshape_16Reshapemul_29Reshape_16/shape*
T0*
Tshape0*#
_output_shapes
:         
S
sub_18Sub
Reshape_15
Reshape_16*
T0*#
_output_shapes
:         
O

norm_6/mulMulsub_18sub_18*
T0*#
_output_shapes
:         
V
norm_6/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
m

norm_6/SumSum
norm_6/mulnorm_6/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
D
norm_6/SqrtSqrt
norm_6/Sum*
T0*
_output_shapes
:
[
norm_6/SqueezeSqueezenorm_6/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
Y
Abs_37Absconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
N
	add_100/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
^
add_100AddV2Abs_37	add_100/y*
T0*0
_output_shapes
:         А
Q
Log_74Logadd_100*
T0*0
_output_shapes
:         А
M
Const_68Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_75LogConst_68*
T0*
_output_shapes
: 
`

truediv_37RealDivLog_74Log_75*
T0*0
_output_shapes
:         А
Q
ReadVariableOp_50ReadVariableOpslope_3*
_output_shapes
: *
dtype0
g
mul_30MulReadVariableOp_50
truediv_37*
T0*0
_output_shapes
:         А
U
ReadVariableOp_51ReadVariableOpintercept_3*
_output_shapes
: *
dtype0
f
add_101AddV2ReadVariableOp_51mul_30*
T0*0
_output_shapes
:         А
Е
differentiable_round_25Roundadd_101*
T0*
_gradient_op_type
Identity*0
_output_shapes
:         А
U
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
|
GreaterEqual_1GreaterEqualconv2d_2/Conv2DGreaterEqual_1/y*
T0*0
_output_shapes
:         А
`
ones_like_2/ShapeShapeconv2d_2/Conv2D*
T0*
_output_shapes
:*
out_type0
V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ж
ones_like_2Fillones_like_2/Shapeones_like_2/Const*
T0*0
_output_shapes
:         А*

index_type0
e
zeros_like_2	ZerosLikeconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
|

SelectV2_2SelectV2GreaterEqual_1ones_like_2zeros_like_2*
T0*0
_output_shapes
:         А
R
LessEqual_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
s
LessEqual_1	LessEqualconv2d_2/Conv2DLessEqual_1/y*
T0*0
_output_shapes
:         А
`
ones_like_3/ShapeShapeconv2d_2/Conv2D*
T0*
_output_shapes
:*
out_type0
V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ж
ones_like_3Fillones_like_3/Shapeones_like_3/Const*
T0*0
_output_shapes
:         А*

index_type0
M
mul_31/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
_
mul_31Mulmul_31/xones_like_3*
T0*0
_output_shapes
:         А
e
zeros_like_3	ZerosLikeconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
t

SelectV2_3SelectV2LessEqual_1mul_31zeros_like_3*
T0*0
_output_shapes
:         А
a
Add_102Add
SelectV2_3
SelectV2_2*
T0*0
_output_shapes
:         А
L
pow_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
i
pow_7Powpow_7/xdifferentiable_round_25*
T0*0
_output_shapes
:         А
X
mul_32MulAdd_102pow_7*
T0*0
_output_shapes
:         А
c
Reshape_17/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
t

Reshape_17Reshapeconv2d_2/Conv2DReshape_17/shape*
T0*
Tshape0*#
_output_shapes
:         
c
Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
k

Reshape_18Reshapemul_32Reshape_18/shape*
T0*
Tshape0*#
_output_shapes
:         
S
sub_19Sub
Reshape_17
Reshape_18*
T0*#
_output_shapes
:         
O

norm_7/mulMulsub_19sub_19*
T0*#
_output_shapes
:         
V
norm_7/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
m

norm_7/SumSum
norm_7/mulnorm_7/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
D
norm_7/SqrtSqrt
norm_7/Sum*
T0*
_output_shapes
:
[
norm_7/SqueezeSqueezenorm_7/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
b

Identity_6Identityconv2d_3/Conv2D*
T0*0
_output_shapes
:         А
c
Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
t

Reshape_19Reshapeconv2d_3/Conv2DReshape_19/shape*
T0*
Tshape0*#
_output_shapes
:         
c
Reshape_20/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
o

Reshape_20Reshape
Identity_6Reshape_20/shape*
T0*
Tshape0*#
_output_shapes
:         
S
sub_20Sub
Reshape_19
Reshape_20*
T0*#
_output_shapes
:         
O

norm_8/mulMulsub_20sub_20*
T0*#
_output_shapes
:         
V
norm_8/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
m

norm_8/SumSum
norm_8/mulnorm_8/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
D
norm_8/SqrtSqrt
norm_8/Sum*
T0*
_output_shapes
:
[
norm_8/SqueezeSqueezenorm_8/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
b

Identity_7Identityconv2d_3/Conv2D*
T0*0
_output_shapes
:         А
c
Reshape_21/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
t

Reshape_21Reshapeconv2d_3/Conv2DReshape_21/shape*
T0*
Tshape0*#
_output_shapes
:         
c
Reshape_22/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
o

Reshape_22Reshape
Identity_7Reshape_22/shape*
T0*
Tshape0*#
_output_shapes
:         
S
sub_21Sub
Reshape_21
Reshape_22*
T0*#
_output_shapes
:         
O

norm_9/mulMulsub_21sub_21*
T0*#
_output_shapes
:         
V
norm_9/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
m

norm_9/SumSum
norm_9/mulnorm_9/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
D
norm_9/SqrtSqrt
norm_9/Sum*
T0*
_output_shapes
:
[
norm_9/SqueezeSqueezenorm_9/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
k
Abs_38/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
U
Abs_38AbsAbs_38/ReadVariableOp*
T0*&
_output_shapes
:
N
	add_103/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_103AddV2Abs_38	add_103/y*
T0*&
_output_shapes
:
G
Log_76Logadd_103*
T0*&
_output_shapes
:
M
Const_69Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_77LogConst_69*
T0*
_output_shapes
: 
V

truediv_38RealDivLog_76Log_77*
T0*&
_output_shapes
:
O
ReadVariableOp_52ReadVariableOpslope*
_output_shapes
: *
dtype0
]
mul_33MulReadVariableOp_52
truediv_38*
T0*&
_output_shapes
:
S
ReadVariableOp_53ReadVariableOp	intercept*
_output_shapes
: *
dtype0
\
add_104AddV2ReadVariableOp_53mul_33*
T0*&
_output_shapes
:
{
differentiable_round_26Roundadd_104*
T0*
_gradient_op_type
Identity*&
_output_shapes
:
s
GreaterEqual_2/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
U
GreaterEqual_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
А
GreaterEqual_2GreaterEqualGreaterEqual_2/ReadVariableOpGreaterEqual_2/y*
T0*&
_output_shapes
:
p
ones_like_4/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
j
ones_like_4/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
|
ones_like_4Fillones_like_4/Shapeones_like_4/Const*
T0*&
_output_shapes
:*

index_type0
q
zeros_like_4Const*&
_output_shapes
:*
dtype0*%
valueB*    
r

SelectV2_4SelectV2GreaterEqual_2ones_like_4zeros_like_4*
T0*&
_output_shapes
:
p
LessEqual_2/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
R
LessEqual_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
t
LessEqual_2	LessEqualLessEqual_2/ReadVariableOpLessEqual_2/y*
T0*&
_output_shapes
:
p
ones_like_5/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
j
ones_like_5/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
V
ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
|
ones_like_5Fillones_like_5/Shapeones_like_5/Const*
T0*&
_output_shapes
:*

index_type0
M
mul_34/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
U
mul_34Mulmul_34/xones_like_5*
T0*&
_output_shapes
:
q
zeros_like_5Const*&
_output_shapes
:*
dtype0*%
valueB*    
j

SelectV2_5SelectV2LessEqual_2mul_34zeros_like_5*
T0*&
_output_shapes
:
W
Add_105Add
SelectV2_5
SelectV2_4*
T0*&
_output_shapes
:
L
pow_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
_
pow_8Powpow_8/xdifferentiable_round_26*
T0*&
_output_shapes
:
N
mul_35MulAdd_105pow_8*
T0*&
_output_shapes
:
o
Reshape_23/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
c
Reshape_23/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
v

Reshape_23ReshapeReshape_23/ReadVariableOpReshape_23/shape*
T0*
Tshape0*
_output_shapes	
:Ц
c
Reshape_24/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
c

Reshape_24Reshapemul_35Reshape_24/shape*
T0*
Tshape0*
_output_shapes	
:Ц
K
sub_22Sub
Reshape_23
Reshape_24*
T0*
_output_shapes	
:Ц
H
norm_10/mulMulsub_22sub_22*
T0*
_output_shapes	
:Ц
W
norm_10/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
p
norm_10/SumSumnorm_10/mulnorm_10/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
F
norm_10/SqrtSqrtnorm_10/Sum*
T0*
_output_shapes
:
]
norm_10/SqueezeSqueezenorm_10/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
k
Abs_39/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
U
Abs_39AbsAbs_39/ReadVariableOp*
T0*&
_output_shapes
:
N
	add_106/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_106AddV2Abs_39	add_106/y*
T0*&
_output_shapes
:
G
Log_78Logadd_106*
T0*&
_output_shapes
:
M
Const_70Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_79LogConst_70*
T0*
_output_shapes
: 
V

truediv_39RealDivLog_78Log_79*
T0*&
_output_shapes
:
O
ReadVariableOp_54ReadVariableOpslope*
_output_shapes
: *
dtype0
]
mul_36MulReadVariableOp_54
truediv_39*
T0*&
_output_shapes
:
S
ReadVariableOp_55ReadVariableOp	intercept*
_output_shapes
: *
dtype0
\
add_107AddV2ReadVariableOp_55mul_36*
T0*&
_output_shapes
:
{
differentiable_round_27Roundadd_107*
T0*
_gradient_op_type
Identity*&
_output_shapes
:
s
GreaterEqual_3/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
U
GreaterEqual_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
А
GreaterEqual_3GreaterEqualGreaterEqual_3/ReadVariableOpGreaterEqual_3/y*
T0*&
_output_shapes
:
p
ones_like_6/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
j
ones_like_6/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
V
ones_like_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
|
ones_like_6Fillones_like_6/Shapeones_like_6/Const*
T0*&
_output_shapes
:*

index_type0
q
zeros_like_6Const*&
_output_shapes
:*
dtype0*%
valueB*    
r

SelectV2_6SelectV2GreaterEqual_3ones_like_6zeros_like_6*
T0*&
_output_shapes
:
p
LessEqual_3/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
R
LessEqual_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
t
LessEqual_3	LessEqualLessEqual_3/ReadVariableOpLessEqual_3/y*
T0*&
_output_shapes
:
p
ones_like_7/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
j
ones_like_7/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
V
ones_like_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
|
ones_like_7Fillones_like_7/Shapeones_like_7/Const*
T0*&
_output_shapes
:*

index_type0
M
mul_37/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
U
mul_37Mulmul_37/xones_like_7*
T0*&
_output_shapes
:
q
zeros_like_7Const*&
_output_shapes
:*
dtype0*%
valueB*    
j

SelectV2_7SelectV2LessEqual_3mul_37zeros_like_7*
T0*&
_output_shapes
:
W
Add_108Add
SelectV2_7
SelectV2_6*
T0*&
_output_shapes
:
L
pow_9/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
_
pow_9Powpow_9/xdifferentiable_round_27*
T0*&
_output_shapes
:
N
mul_38MulAdd_108pow_9*
T0*&
_output_shapes
:
o
Reshape_25/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
c
Reshape_25/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
v

Reshape_25ReshapeReshape_25/ReadVariableOpReshape_25/shape*
T0*
Tshape0*
_output_shapes	
:Ц
c
Reshape_26/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
c

Reshape_26Reshapemul_38Reshape_26/shape*
T0*
Tshape0*
_output_shapes	
:Ц
K
sub_23Sub
Reshape_25
Reshape_26*
T0*
_output_shapes	
:Ц
H
norm_11/mulMulsub_23sub_23*
T0*
_output_shapes	
:Ц
W
norm_11/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
p
norm_11/SumSumnorm_11/mulnorm_11/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
F
norm_11/SqrtSqrtnorm_11/Sum*
T0*
_output_shapes
:
]
norm_11/SqueezeSqueezenorm_11/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
m
Abs_40/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
U
Abs_40AbsAbs_40/ReadVariableOp*
T0*&
_output_shapes
:

N
	add_109/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_109AddV2Abs_40	add_109/y*
T0*&
_output_shapes
:

G
Log_80Logadd_109*
T0*&
_output_shapes
:

M
Const_71Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_81LogConst_71*
T0*
_output_shapes
: 
V

truediv_40RealDivLog_80Log_81*
T0*&
_output_shapes
:

Q
ReadVariableOp_56ReadVariableOpslope_1*
_output_shapes
: *
dtype0
]
mul_39MulReadVariableOp_56
truediv_40*
T0*&
_output_shapes
:

U
ReadVariableOp_57ReadVariableOpintercept_1*
_output_shapes
: *
dtype0
\
add_110AddV2ReadVariableOp_57mul_39*
T0*&
_output_shapes
:

{
differentiable_round_28Roundadd_110*
T0*
_gradient_op_type
Identity*&
_output_shapes
:

u
GreaterEqual_4/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
U
GreaterEqual_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
А
GreaterEqual_4GreaterEqualGreaterEqual_4/ReadVariableOpGreaterEqual_4/y*
T0*&
_output_shapes
:

r
ones_like_8/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
j
ones_like_8/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
V
ones_like_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
|
ones_like_8Fillones_like_8/Shapeones_like_8/Const*
T0*&
_output_shapes
:
*

index_type0
u
zeros_like_8/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"         
   
W
zeros_like_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Й
zeros_like_8Fillzeros_like_8/shape_as_tensorzeros_like_8/Const*
T0*&
_output_shapes
:
*

index_type0
r

SelectV2_8SelectV2GreaterEqual_4ones_like_8zeros_like_8*
T0*&
_output_shapes
:

r
LessEqual_4/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
R
LessEqual_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
t
LessEqual_4	LessEqualLessEqual_4/ReadVariableOpLessEqual_4/y*
T0*&
_output_shapes
:

r
ones_like_9/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
j
ones_like_9/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
V
ones_like_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
|
ones_like_9Fillones_like_9/Shapeones_like_9/Const*
T0*&
_output_shapes
:
*

index_type0
M
mul_40/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
U
mul_40Mulmul_40/xones_like_9*
T0*&
_output_shapes
:

u
zeros_like_9/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"         
   
W
zeros_like_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Й
zeros_like_9Fillzeros_like_9/shape_as_tensorzeros_like_9/Const*
T0*&
_output_shapes
:
*

index_type0
j

SelectV2_9SelectV2LessEqual_4mul_40zeros_like_9*
T0*&
_output_shapes
:

W
Add_111Add
SelectV2_9
SelectV2_8*
T0*&
_output_shapes
:

M
pow_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
a
pow_10Powpow_10/xdifferentiable_round_28*
T0*&
_output_shapes
:

O
mul_41MulAdd_111pow_10*
T0*&
_output_shapes
:

q
Reshape_27/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
c
Reshape_27/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
v

Reshape_27ReshapeReshape_27/ReadVariableOpReshape_27/shape*
T0*
Tshape0*
_output_shapes	
:▄
c
Reshape_28/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
c

Reshape_28Reshapemul_41Reshape_28/shape*
T0*
Tshape0*
_output_shapes	
:▄
K
sub_24Sub
Reshape_27
Reshape_28*
T0*
_output_shapes	
:▄
H
norm_12/mulMulsub_24sub_24*
T0*
_output_shapes	
:▄
W
norm_12/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
p
norm_12/SumSumnorm_12/mulnorm_12/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
F
norm_12/SqrtSqrtnorm_12/Sum*
T0*
_output_shapes
:
]
norm_12/SqueezeSqueezenorm_12/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
m
Abs_41/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
U
Abs_41AbsAbs_41/ReadVariableOp*
T0*&
_output_shapes
:

N
	add_112/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_112AddV2Abs_41	add_112/y*
T0*&
_output_shapes
:

G
Log_82Logadd_112*
T0*&
_output_shapes
:

M
Const_72Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_83LogConst_72*
T0*
_output_shapes
: 
V

truediv_41RealDivLog_82Log_83*
T0*&
_output_shapes
:

Q
ReadVariableOp_58ReadVariableOpslope_1*
_output_shapes
: *
dtype0
]
mul_42MulReadVariableOp_58
truediv_41*
T0*&
_output_shapes
:

U
ReadVariableOp_59ReadVariableOpintercept_1*
_output_shapes
: *
dtype0
\
add_113AddV2ReadVariableOp_59mul_42*
T0*&
_output_shapes
:

{
differentiable_round_29Roundadd_113*
T0*
_gradient_op_type
Identity*&
_output_shapes
:

u
GreaterEqual_5/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
U
GreaterEqual_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
А
GreaterEqual_5GreaterEqualGreaterEqual_5/ReadVariableOpGreaterEqual_5/y*
T0*&
_output_shapes
:

s
ones_like_10/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
k
ones_like_10/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
W
ones_like_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?

ones_like_10Fillones_like_10/Shapeones_like_10/Const*
T0*&
_output_shapes
:
*

index_type0
v
zeros_like_10/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"         
   
X
zeros_like_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
М
zeros_like_10Fillzeros_like_10/shape_as_tensorzeros_like_10/Const*
T0*&
_output_shapes
:
*

index_type0
u
SelectV2_10SelectV2GreaterEqual_5ones_like_10zeros_like_10*
T0*&
_output_shapes
:

r
LessEqual_5/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
R
LessEqual_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
t
LessEqual_5	LessEqualLessEqual_5/ReadVariableOpLessEqual_5/y*
T0*&
_output_shapes
:

s
ones_like_11/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
k
ones_like_11/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
W
ones_like_11/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?

ones_like_11Fillones_like_11/Shapeones_like_11/Const*
T0*&
_output_shapes
:
*

index_type0
M
mul_43/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
V
mul_43Mulmul_43/xones_like_11*
T0*&
_output_shapes
:

v
zeros_like_11/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"         
   
X
zeros_like_11/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
М
zeros_like_11Fillzeros_like_11/shape_as_tensorzeros_like_11/Const*
T0*&
_output_shapes
:
*

index_type0
l
SelectV2_11SelectV2LessEqual_5mul_43zeros_like_11*
T0*&
_output_shapes
:

Y
Add_114AddSelectV2_11SelectV2_10*
T0*&
_output_shapes
:

M
pow_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
a
pow_11Powpow_11/xdifferentiable_round_29*
T0*&
_output_shapes
:

O
mul_44MulAdd_114pow_11*
T0*&
_output_shapes
:

q
Reshape_29/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
c
Reshape_29/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
v

Reshape_29ReshapeReshape_29/ReadVariableOpReshape_29/shape*
T0*
Tshape0*
_output_shapes	
:▄
c
Reshape_30/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
c

Reshape_30Reshapemul_44Reshape_30/shape*
T0*
Tshape0*
_output_shapes	
:▄
K
sub_25Sub
Reshape_29
Reshape_30*
T0*
_output_shapes	
:▄
H
norm_13/mulMulsub_25sub_25*
T0*
_output_shapes	
:▄
W
norm_13/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
p
norm_13/SumSumnorm_13/mulnorm_13/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
F
norm_13/SqrtSqrtnorm_13/Sum*
T0*
_output_shapes
:
]
norm_13/SqueezeSqueezenorm_13/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
n
Abs_42/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
V
Abs_42AbsAbs_42/ReadVariableOp*
T0*'
_output_shapes
:1
А
N
	add_115/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
U
add_115AddV2Abs_42	add_115/y*
T0*'
_output_shapes
:1
А
H
Log_84Logadd_115*
T0*'
_output_shapes
:1
А
M
Const_73Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_85LogConst_73*
T0*
_output_shapes
: 
W

truediv_42RealDivLog_84Log_85*
T0*'
_output_shapes
:1
А
Q
ReadVariableOp_60ReadVariableOpslope_2*
_output_shapes
: *
dtype0
^
mul_45MulReadVariableOp_60
truediv_42*
T0*'
_output_shapes
:1
А
U
ReadVariableOp_61ReadVariableOpintercept_2*
_output_shapes
: *
dtype0
]
add_116AddV2ReadVariableOp_61mul_45*
T0*'
_output_shapes
:1
А
|
differentiable_round_30Roundadd_116*
T0*
_gradient_op_type
Identity*'
_output_shapes
:1
А
v
GreaterEqual_6/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
U
GreaterEqual_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Б
GreaterEqual_6GreaterEqualGreaterEqual_6/ReadVariableOpGreaterEqual_6/y*
T0*'
_output_shapes
:1
А
t
ones_like_12/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
k
ones_like_12/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
W
ones_like_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_12Fillones_like_12/Shapeones_like_12/Const*
T0*'
_output_shapes
:1
А*

index_type0
v
zeros_like_12/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
X
zeros_like_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_12Fillzeros_like_12/shape_as_tensorzeros_like_12/Const*
T0*'
_output_shapes
:1
А*

index_type0
v
SelectV2_12SelectV2GreaterEqual_6ones_like_12zeros_like_12*
T0*'
_output_shapes
:1
А
s
LessEqual_6/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
R
LessEqual_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
u
LessEqual_6	LessEqualLessEqual_6/ReadVariableOpLessEqual_6/y*
T0*'
_output_shapes
:1
А
t
ones_like_13/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
k
ones_like_13/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
W
ones_like_13/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_13Fillones_like_13/Shapeones_like_13/Const*
T0*'
_output_shapes
:1
А*

index_type0
M
mul_46/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
W
mul_46Mulmul_46/xones_like_13*
T0*'
_output_shapes
:1
А
v
zeros_like_13/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
X
zeros_like_13/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_13Fillzeros_like_13/shape_as_tensorzeros_like_13/Const*
T0*'
_output_shapes
:1
А*

index_type0
m
SelectV2_13SelectV2LessEqual_6mul_46zeros_like_13*
T0*'
_output_shapes
:1
А
Z
Add_117AddSelectV2_13SelectV2_12*
T0*'
_output_shapes
:1
А
M
pow_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
b
pow_12Powpow_12/xdifferentiable_round_30*
T0*'
_output_shapes
:1
А
P
mul_47MulAdd_117pow_12*
T0*'
_output_shapes
:1
А
r
Reshape_31/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
c
Reshape_31/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
w

Reshape_31ReshapeReshape_31/ReadVariableOpReshape_31/shape*
T0*
Tshape0*
_output_shapes

:Аъ
c
Reshape_32/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
d

Reshape_32Reshapemul_47Reshape_32/shape*
T0*
Tshape0*
_output_shapes

:Аъ
L
sub_26Sub
Reshape_31
Reshape_32*
T0*
_output_shapes

:Аъ
I
norm_14/mulMulsub_26sub_26*
T0*
_output_shapes

:Аъ
W
norm_14/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
p
norm_14/SumSumnorm_14/mulnorm_14/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
F
norm_14/SqrtSqrtnorm_14/Sum*
T0*
_output_shapes
:
]
norm_14/SqueezeSqueezenorm_14/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
n
Abs_43/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
V
Abs_43AbsAbs_43/ReadVariableOp*
T0*'
_output_shapes
:1
А
N
	add_118/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
U
add_118AddV2Abs_43	add_118/y*
T0*'
_output_shapes
:1
А
H
Log_86Logadd_118*
T0*'
_output_shapes
:1
А
M
Const_74Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_87LogConst_74*
T0*
_output_shapes
: 
W

truediv_43RealDivLog_86Log_87*
T0*'
_output_shapes
:1
А
Q
ReadVariableOp_62ReadVariableOpslope_2*
_output_shapes
: *
dtype0
^
mul_48MulReadVariableOp_62
truediv_43*
T0*'
_output_shapes
:1
А
U
ReadVariableOp_63ReadVariableOpintercept_2*
_output_shapes
: *
dtype0
]
add_119AddV2ReadVariableOp_63mul_48*
T0*'
_output_shapes
:1
А
|
differentiable_round_31Roundadd_119*
T0*
_gradient_op_type
Identity*'
_output_shapes
:1
А
v
GreaterEqual_7/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
U
GreaterEqual_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Б
GreaterEqual_7GreaterEqualGreaterEqual_7/ReadVariableOpGreaterEqual_7/y*
T0*'
_output_shapes
:1
А
t
ones_like_14/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
k
ones_like_14/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
W
ones_like_14/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_14Fillones_like_14/Shapeones_like_14/Const*
T0*'
_output_shapes
:1
А*

index_type0
v
zeros_like_14/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
X
zeros_like_14/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_14Fillzeros_like_14/shape_as_tensorzeros_like_14/Const*
T0*'
_output_shapes
:1
А*

index_type0
v
SelectV2_14SelectV2GreaterEqual_7ones_like_14zeros_like_14*
T0*'
_output_shapes
:1
А
s
LessEqual_7/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
R
LessEqual_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
u
LessEqual_7	LessEqualLessEqual_7/ReadVariableOpLessEqual_7/y*
T0*'
_output_shapes
:1
А
t
ones_like_15/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
k
ones_like_15/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
W
ones_like_15/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_15Fillones_like_15/Shapeones_like_15/Const*
T0*'
_output_shapes
:1
А*

index_type0
M
mul_49/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
W
mul_49Mulmul_49/xones_like_15*
T0*'
_output_shapes
:1
А
v
zeros_like_15/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
X
zeros_like_15/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_15Fillzeros_like_15/shape_as_tensorzeros_like_15/Const*
T0*'
_output_shapes
:1
А*

index_type0
m
SelectV2_15SelectV2LessEqual_7mul_49zeros_like_15*
T0*'
_output_shapes
:1
А
Z
Add_120AddSelectV2_15SelectV2_14*
T0*'
_output_shapes
:1
А
M
pow_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
b
pow_13Powpow_13/xdifferentiable_round_31*
T0*'
_output_shapes
:1
А
P
mul_50MulAdd_120pow_13*
T0*'
_output_shapes
:1
А
r
Reshape_33/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
c
Reshape_33/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
w

Reshape_33ReshapeReshape_33/ReadVariableOpReshape_33/shape*
T0*
Tshape0*
_output_shapes

:Аъ
c
Reshape_34/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
d

Reshape_34Reshapemul_50Reshape_34/shape*
T0*
Tshape0*
_output_shapes

:Аъ
L
sub_27Sub
Reshape_33
Reshape_34*
T0*
_output_shapes

:Аъ
I
norm_15/mulMulsub_27sub_27*
T0*
_output_shapes

:Аъ
W
norm_15/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
p
norm_15/SumSumnorm_15/mulnorm_15/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
F
norm_15/SqrtSqrtnorm_15/Sum*
T0*
_output_shapes
:
]
norm_15/SqueezeSqueezenorm_15/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
o
Abs_44/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
W
Abs_44AbsAbs_44/ReadVariableOp*
T0*(
_output_shapes
:АА
N
	add_121/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
V
add_121AddV2Abs_44	add_121/y*
T0*(
_output_shapes
:АА
I
Log_88Logadd_121*
T0*(
_output_shapes
:АА
M
Const_75Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_89LogConst_75*
T0*
_output_shapes
: 
X

truediv_44RealDivLog_88Log_89*
T0*(
_output_shapes
:АА
Q
ReadVariableOp_64ReadVariableOpslope_4*
_output_shapes
: *
dtype0
_
mul_51MulReadVariableOp_64
truediv_44*
T0*(
_output_shapes
:АА
U
ReadVariableOp_65ReadVariableOpintercept_4*
_output_shapes
: *
dtype0
^
add_122AddV2ReadVariableOp_65mul_51*
T0*(
_output_shapes
:АА
}
differentiable_round_32Roundadd_122*
T0*
_gradient_op_type
Identity*(
_output_shapes
:АА
w
GreaterEqual_8/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
U
GreaterEqual_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
В
GreaterEqual_8GreaterEqualGreaterEqual_8/ReadVariableOpGreaterEqual_8/y*
T0*(
_output_shapes
:АА
u
ones_like_16/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
k
ones_like_16/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
W
ones_like_16/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Б
ones_like_16Fillones_like_16/Shapeones_like_16/Const*
T0*(
_output_shapes
:АА*

index_type0
v
zeros_like_16/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
X
zeros_like_16/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
О
zeros_like_16Fillzeros_like_16/shape_as_tensorzeros_like_16/Const*
T0*(
_output_shapes
:АА*

index_type0
w
SelectV2_16SelectV2GreaterEqual_8ones_like_16zeros_like_16*
T0*(
_output_shapes
:АА
t
LessEqual_8/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
R
LessEqual_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
v
LessEqual_8	LessEqualLessEqual_8/ReadVariableOpLessEqual_8/y*
T0*(
_output_shapes
:АА
u
ones_like_17/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
k
ones_like_17/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
W
ones_like_17/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Б
ones_like_17Fillones_like_17/Shapeones_like_17/Const*
T0*(
_output_shapes
:АА*

index_type0
M
mul_52/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
X
mul_52Mulmul_52/xones_like_17*
T0*(
_output_shapes
:АА
v
zeros_like_17/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
X
zeros_like_17/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
О
zeros_like_17Fillzeros_like_17/shape_as_tensorzeros_like_17/Const*
T0*(
_output_shapes
:АА*

index_type0
n
SelectV2_17SelectV2LessEqual_8mul_52zeros_like_17*
T0*(
_output_shapes
:АА
[
Add_123AddSelectV2_17SelectV2_16*
T0*(
_output_shapes
:АА
M
pow_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
c
pow_14Powpow_14/xdifferentiable_round_32*
T0*(
_output_shapes
:АА
Q
mul_53MulAdd_123pow_14*
T0*(
_output_shapes
:АА
s
Reshape_35/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
c
Reshape_35/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
w

Reshape_35ReshapeReshape_35/ReadVariableOpReshape_35/shape*
T0*
Tshape0*
_output_shapes

:АА
c
Reshape_36/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
d

Reshape_36Reshapemul_53Reshape_36/shape*
T0*
Tshape0*
_output_shapes

:АА
L
sub_28Sub
Reshape_35
Reshape_36*
T0*
_output_shapes

:АА
I
norm_16/mulMulsub_28sub_28*
T0*
_output_shapes

:АА
W
norm_16/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
p
norm_16/SumSumnorm_16/mulnorm_16/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
F
norm_16/SqrtSqrtnorm_16/Sum*
T0*
_output_shapes
:
]
norm_16/SqueezeSqueezenorm_16/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
o
Abs_45/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
W
Abs_45AbsAbs_45/ReadVariableOp*
T0*(
_output_shapes
:АА
N
	add_124/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
V
add_124AddV2Abs_45	add_124/y*
T0*(
_output_shapes
:АА
I
Log_90Logadd_124*
T0*(
_output_shapes
:АА
M
Const_76Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_91LogConst_76*
T0*
_output_shapes
: 
X

truediv_45RealDivLog_90Log_91*
T0*(
_output_shapes
:АА
Q
ReadVariableOp_66ReadVariableOpslope_4*
_output_shapes
: *
dtype0
_
mul_54MulReadVariableOp_66
truediv_45*
T0*(
_output_shapes
:АА
U
ReadVariableOp_67ReadVariableOpintercept_4*
_output_shapes
: *
dtype0
^
add_125AddV2ReadVariableOp_67mul_54*
T0*(
_output_shapes
:АА
}
differentiable_round_33Roundadd_125*
T0*
_gradient_op_type
Identity*(
_output_shapes
:АА
w
GreaterEqual_9/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
U
GreaterEqual_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
В
GreaterEqual_9GreaterEqualGreaterEqual_9/ReadVariableOpGreaterEqual_9/y*
T0*(
_output_shapes
:АА
u
ones_like_18/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
k
ones_like_18/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
W
ones_like_18/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Б
ones_like_18Fillones_like_18/Shapeones_like_18/Const*
T0*(
_output_shapes
:АА*

index_type0
v
zeros_like_18/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
X
zeros_like_18/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
О
zeros_like_18Fillzeros_like_18/shape_as_tensorzeros_like_18/Const*
T0*(
_output_shapes
:АА*

index_type0
w
SelectV2_18SelectV2GreaterEqual_9ones_like_18zeros_like_18*
T0*(
_output_shapes
:АА
t
LessEqual_9/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
R
LessEqual_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
v
LessEqual_9	LessEqualLessEqual_9/ReadVariableOpLessEqual_9/y*
T0*(
_output_shapes
:АА
u
ones_like_19/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
k
ones_like_19/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
W
ones_like_19/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Б
ones_like_19Fillones_like_19/Shapeones_like_19/Const*
T0*(
_output_shapes
:АА*

index_type0
M
mul_55/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
X
mul_55Mulmul_55/xones_like_19*
T0*(
_output_shapes
:АА
v
zeros_like_19/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
X
zeros_like_19/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
О
zeros_like_19Fillzeros_like_19/shape_as_tensorzeros_like_19/Const*
T0*(
_output_shapes
:АА*

index_type0
n
SelectV2_19SelectV2LessEqual_9mul_55zeros_like_19*
T0*(
_output_shapes
:АА
[
Add_126AddSelectV2_19SelectV2_18*
T0*(
_output_shapes
:АА
M
pow_15/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
c
pow_15Powpow_15/xdifferentiable_round_33*
T0*(
_output_shapes
:АА
Q
mul_56MulAdd_126pow_15*
T0*(
_output_shapes
:АА
s
Reshape_37/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
c
Reshape_37/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
w

Reshape_37ReshapeReshape_37/ReadVariableOpReshape_37/shape*
T0*
Tshape0*
_output_shapes

:АА
c
Reshape_38/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
d

Reshape_38Reshapemul_56Reshape_38/shape*
T0*
Tshape0*
_output_shapes

:АА
L
sub_29Sub
Reshape_37
Reshape_38*
T0*
_output_shapes

:АА
I
norm_17/mulMulsub_29sub_29*
T0*
_output_shapes

:АА
W
norm_17/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
p
norm_17/SumSumnorm_17/mulnorm_17/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
F
norm_17/SqrtSqrtnorm_17/Sum*
T0*
_output_shapes
:
]
norm_17/SqueezeSqueezenorm_17/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
n
Abs_46/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
V
Abs_46AbsAbs_46/ReadVariableOp*
T0*'
_output_shapes
:А

N
	add_127/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
U
add_127AddV2Abs_46	add_127/y*
T0*'
_output_shapes
:А

H
Log_92Logadd_127*
T0*'
_output_shapes
:А

M
Const_77Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_93LogConst_77*
T0*
_output_shapes
: 
W

truediv_46RealDivLog_92Log_93*
T0*'
_output_shapes
:А

Q
ReadVariableOp_68ReadVariableOpslope_5*
_output_shapes
: *
dtype0
^
mul_57MulReadVariableOp_68
truediv_46*
T0*'
_output_shapes
:А

U
ReadVariableOp_69ReadVariableOpintercept_5*
_output_shapes
: *
dtype0
]
add_128AddV2ReadVariableOp_69mul_57*
T0*'
_output_shapes
:А

|
differentiable_round_34Roundadd_128*
T0*
_gradient_op_type
Identity*'
_output_shapes
:А

w
GreaterEqual_10/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
V
GreaterEqual_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Д
GreaterEqual_10GreaterEqualGreaterEqual_10/ReadVariableOpGreaterEqual_10/y*
T0*'
_output_shapes
:А

t
ones_like_20/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
k
ones_like_20/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
W
ones_like_20/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_20Fillones_like_20/Shapeones_like_20/Const*
T0*'
_output_shapes
:А
*

index_type0
v
zeros_like_20/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
X
zeros_like_20/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_20Fillzeros_like_20/shape_as_tensorzeros_like_20/Const*
T0*'
_output_shapes
:А
*

index_type0
w
SelectV2_20SelectV2GreaterEqual_10ones_like_20zeros_like_20*
T0*'
_output_shapes
:А

t
LessEqual_10/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
S
LessEqual_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
x
LessEqual_10	LessEqualLessEqual_10/ReadVariableOpLessEqual_10/y*
T0*'
_output_shapes
:А

t
ones_like_21/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
k
ones_like_21/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
W
ones_like_21/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_21Fillones_like_21/Shapeones_like_21/Const*
T0*'
_output_shapes
:А
*

index_type0
M
mul_58/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
W
mul_58Mulmul_58/xones_like_21*
T0*'
_output_shapes
:А

v
zeros_like_21/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
X
zeros_like_21/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_21Fillzeros_like_21/shape_as_tensorzeros_like_21/Const*
T0*'
_output_shapes
:А
*

index_type0
n
SelectV2_21SelectV2LessEqual_10mul_58zeros_like_21*
T0*'
_output_shapes
:А

Z
Add_129AddSelectV2_21SelectV2_20*
T0*'
_output_shapes
:А

M
pow_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
b
pow_16Powpow_16/xdifferentiable_round_34*
T0*'
_output_shapes
:А

P
mul_59MulAdd_129pow_16*
T0*'
_output_shapes
:А

r
Reshape_39/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
c
Reshape_39/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
v

Reshape_39ReshapeReshape_39/ReadVariableOpReshape_39/shape*
T0*
Tshape0*
_output_shapes	
:А

c
Reshape_40/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
c

Reshape_40Reshapemul_59Reshape_40/shape*
T0*
Tshape0*
_output_shapes	
:А

K
sub_30Sub
Reshape_39
Reshape_40*
T0*
_output_shapes	
:А

H
norm_18/mulMulsub_30sub_30*
T0*
_output_shapes	
:А

W
norm_18/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
p
norm_18/SumSumnorm_18/mulnorm_18/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
F
norm_18/SqrtSqrtnorm_18/Sum*
T0*
_output_shapes
:
]
norm_18/SqueezeSqueezenorm_18/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
n
Abs_47/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
V
Abs_47AbsAbs_47/ReadVariableOp*
T0*'
_output_shapes
:А

N
	add_130/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
U
add_130AddV2Abs_47	add_130/y*
T0*'
_output_shapes
:А

H
Log_94Logadd_130*
T0*'
_output_shapes
:А

M
Const_78Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_95LogConst_78*
T0*
_output_shapes
: 
W

truediv_47RealDivLog_94Log_95*
T0*'
_output_shapes
:А

Q
ReadVariableOp_70ReadVariableOpslope_5*
_output_shapes
: *
dtype0
^
mul_60MulReadVariableOp_70
truediv_47*
T0*'
_output_shapes
:А

U
ReadVariableOp_71ReadVariableOpintercept_5*
_output_shapes
: *
dtype0
]
add_131AddV2ReadVariableOp_71mul_60*
T0*'
_output_shapes
:А

|
differentiable_round_35Roundadd_131*
T0*
_gradient_op_type
Identity*'
_output_shapes
:А

w
GreaterEqual_11/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
V
GreaterEqual_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Д
GreaterEqual_11GreaterEqualGreaterEqual_11/ReadVariableOpGreaterEqual_11/y*
T0*'
_output_shapes
:А

t
ones_like_22/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
k
ones_like_22/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
W
ones_like_22/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_22Fillones_like_22/Shapeones_like_22/Const*
T0*'
_output_shapes
:А
*

index_type0
v
zeros_like_22/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
X
zeros_like_22/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_22Fillzeros_like_22/shape_as_tensorzeros_like_22/Const*
T0*'
_output_shapes
:А
*

index_type0
w
SelectV2_22SelectV2GreaterEqual_11ones_like_22zeros_like_22*
T0*'
_output_shapes
:А

t
LessEqual_11/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
S
LessEqual_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
x
LessEqual_11	LessEqualLessEqual_11/ReadVariableOpLessEqual_11/y*
T0*'
_output_shapes
:А

t
ones_like_23/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
k
ones_like_23/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
W
ones_like_23/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_23Fillones_like_23/Shapeones_like_23/Const*
T0*'
_output_shapes
:А
*

index_type0
M
mul_61/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
W
mul_61Mulmul_61/xones_like_23*
T0*'
_output_shapes
:А

v
zeros_like_23/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
X
zeros_like_23/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_23Fillzeros_like_23/shape_as_tensorzeros_like_23/Const*
T0*'
_output_shapes
:А
*

index_type0
n
SelectV2_23SelectV2LessEqual_11mul_61zeros_like_23*
T0*'
_output_shapes
:А

Z
Add_132AddSelectV2_23SelectV2_22*
T0*'
_output_shapes
:А

M
pow_17/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
b
pow_17Powpow_17/xdifferentiable_round_35*
T0*'
_output_shapes
:А

P
mul_62MulAdd_132pow_17*
T0*'
_output_shapes
:А

r
Reshape_41/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
c
Reshape_41/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
v

Reshape_41ReshapeReshape_41/ReadVariableOpReshape_41/shape*
T0*
Tshape0*
_output_shapes	
:А

c
Reshape_42/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
c

Reshape_42Reshapemul_62Reshape_42/shape*
T0*
Tshape0*
_output_shapes	
:А

K
sub_31Sub
Reshape_41
Reshape_42*
T0*
_output_shapes	
:А

H
norm_19/mulMulsub_31sub_31*
T0*
_output_shapes	
:А

W
norm_19/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
p
norm_19/SumSumnorm_19/mulnorm_19/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
F
norm_19/SqrtSqrtnorm_19/Sum*
T0*
_output_shapes
:
]
norm_19/SqueezeSqueezenorm_19/Sqrt*
T0*
_output_shapes
: *
squeeze_dims
 
]

Identity_8IdentityPlaceholder*
T0*/
_output_shapes
:         
c
Reshape_43/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
p

Reshape_43ReshapePlaceholderReshape_43/shape*
T0*
Tshape0*#
_output_shapes
:         
W
l2_normalize/SquareSquare
Reshape_43*
T0*#
_output_shapes
:         
\
l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
В
l2_normalize/SumSuml2_normalize/Squarel2_normalize/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
[
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
n
l2_normalize/MaximumMaximuml2_normalize/Suml2_normalize/Maximum/y*
T0*
_output_shapes
:
V
l2_normalize/RsqrtRsqrtl2_normalize/Maximum*
T0*
_output_shapes
:
a
l2_normalizeMul
Reshape_43l2_normalize/Rsqrt*
T0*#
_output_shapes
:         
c
Reshape_44/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
o

Reshape_44Reshape
Identity_8Reshape_44/shape*
T0*
Tshape0*#
_output_shapes
:         
Y
l2_normalize_1/SquareSquare
Reshape_44*
T0*#
_output_shapes
:         
^
l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
И
l2_normalize_1/SumSuml2_normalize_1/Squarel2_normalize_1/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
t
l2_normalize_1/MaximumMaximuml2_normalize_1/Suml2_normalize_1/Maximum/y*
T0*
_output_shapes
:
Z
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum*
T0*
_output_shapes
:
e
l2_normalize_1Mul
Reshape_44l2_normalize_1/Rsqrt*
T0*#
_output_shapes
:         
X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB: 
Q
Tensordot/freeConst*
_output_shapes
: *
dtype0*
valueB 
[
Tensordot/ShapeShapel2_normalize*
T0*
_output_shapes
:*
out_type0
Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
░
Tensordot/GatherV2GatherV2Tensordot/ShapeTensordot/freeTensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╢
Tensordot/GatherV2_1GatherV2Tensordot/ShapeTensordot/axesTensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
y
Tensordot/ProdProdTensordot/GatherV2Tensordot/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
[
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 

Tensordot/Prod_1ProdTensordot/GatherV2_1Tensordot/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Н
Tensordot/concatConcatV2Tensordot/freeTensordot/axesTensordot/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
s
Tensordot/stackPackTensordot/ProdTensordot/Prod_1*
N*
T0*
_output_shapes
:*

axis 
{
Tensordot/transpose	Transposel2_normalizeTensordot/concat*
T0*
Tperm0*#
_output_shapes
:         
Л
Tensordot/ReshapeReshapeTensordot/transposeTensordot/stack*
T0*
Tshape0*0
_output_shapes
:                  
Z
Tensordot/axes_1Const*
_output_shapes
:*
dtype0*
valueB: 
S
Tensordot/free_1Const*
_output_shapes
: *
dtype0*
valueB 
_
Tensordot/Shape_1Shapel2_normalize_1*
T0*
_output_shapes
:*
out_type0
[
Tensordot/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╕
Tensordot/GatherV2_2GatherV2Tensordot/Shape_1Tensordot/free_1Tensordot/GatherV2_2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
[
Tensordot/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
║
Tensordot/GatherV2_3GatherV2Tensordot/Shape_1Tensordot/axes_1Tensordot/GatherV2_3/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 

Tensordot/Prod_2ProdTensordot/GatherV2_2Tensordot/Const_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
[
Tensordot/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 

Tensordot/Prod_3ProdTensordot/GatherV2_3Tensordot/Const_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Х
Tensordot/concat_1ConcatV2Tensordot/axes_1Tensordot/free_1Tensordot/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
w
Tensordot/stack_1PackTensordot/Prod_3Tensordot/Prod_2*
N*
T0*
_output_shapes
:*

axis 
Б
Tensordot/transpose_1	Transposel2_normalize_1Tensordot/concat_1*
T0*
Tperm0*#
_output_shapes
:         
С
Tensordot/Reshape_1ReshapeTensordot/transpose_1Tensordot/stack_1*
T0*
Tshape0*0
_output_shapes
:                  
г
Tensordot/MatMulMatMulTensordot/ReshapeTensordot/Reshape_1*
T0*0
_output_shapes
:                  *
transpose_a( *
transpose_b( 
Y
Tensordot/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Щ
Tensordot/concat_2ConcatV2Tensordot/GatherV2Tensordot/GatherV2_2Tensordot/concat_2/axis*
N*
T0*

Tidx0*
_output_shapes
: 
i
	TensordotReshapeTensordot/MatMulTensordot/concat_2*
T0*
Tshape0*
_output_shapes
: 
]

Identity_9IdentityPlaceholder*
T0*/
_output_shapes
:         
c
Reshape_45/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
p

Reshape_45ReshapePlaceholderReshape_45/shape*
T0*
Tshape0*#
_output_shapes
:         
Y
l2_normalize_2/SquareSquare
Reshape_45*
T0*#
_output_shapes
:         
^
l2_normalize_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
И
l2_normalize_2/SumSuml2_normalize_2/Squarel2_normalize_2/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
]
l2_normalize_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
t
l2_normalize_2/MaximumMaximuml2_normalize_2/Suml2_normalize_2/Maximum/y*
T0*
_output_shapes
:
Z
l2_normalize_2/RsqrtRsqrtl2_normalize_2/Maximum*
T0*
_output_shapes
:
e
l2_normalize_2Mul
Reshape_45l2_normalize_2/Rsqrt*
T0*#
_output_shapes
:         
c
Reshape_46/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
o

Reshape_46Reshape
Identity_9Reshape_46/shape*
T0*
Tshape0*#
_output_shapes
:         
Y
l2_normalize_3/SquareSquare
Reshape_46*
T0*#
_output_shapes
:         
^
l2_normalize_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
И
l2_normalize_3/SumSuml2_normalize_3/Squarel2_normalize_3/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
]
l2_normalize_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
t
l2_normalize_3/MaximumMaximuml2_normalize_3/Suml2_normalize_3/Maximum/y*
T0*
_output_shapes
:
Z
l2_normalize_3/RsqrtRsqrtl2_normalize_3/Maximum*
T0*
_output_shapes
:
e
l2_normalize_3Mul
Reshape_46l2_normalize_3/Rsqrt*
T0*#
_output_shapes
:         
Z
Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB: 
S
Tensordot_1/freeConst*
_output_shapes
: *
dtype0*
valueB 
_
Tensordot_1/ShapeShapel2_normalize_2*
T0*
_output_shapes
:*
out_type0
[
Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╕
Tensordot_1/GatherV2GatherV2Tensordot_1/ShapeTensordot_1/freeTensordot_1/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╛
Tensordot_1/GatherV2_1GatherV2Tensordot_1/ShapeTensordot_1/axesTensordot_1/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
[
Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

Tensordot_1/ProdProdTensordot_1/GatherV2Tensordot_1/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_1/Prod_1ProdTensordot_1/GatherV2_1Tensordot_1/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Y
Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Х
Tensordot_1/concatConcatV2Tensordot_1/freeTensordot_1/axesTensordot_1/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
y
Tensordot_1/stackPackTensordot_1/ProdTensordot_1/Prod_1*
N*
T0*
_output_shapes
:*

axis 
Б
Tensordot_1/transpose	Transposel2_normalize_2Tensordot_1/concat*
T0*
Tperm0*#
_output_shapes
:         
С
Tensordot_1/ReshapeReshapeTensordot_1/transposeTensordot_1/stack*
T0*
Tshape0*0
_output_shapes
:                  
\
Tensordot_1/axes_1Const*
_output_shapes
:*
dtype0*
valueB: 
U
Tensordot_1/free_1Const*
_output_shapes
: *
dtype0*
valueB 
a
Tensordot_1/Shape_1Shapel2_normalize_3*
T0*
_output_shapes
:*
out_type0
]
Tensordot_1/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
└
Tensordot_1/GatherV2_2GatherV2Tensordot_1/Shape_1Tensordot_1/free_1Tensordot_1/GatherV2_2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_1/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
┬
Tensordot_1/GatherV2_3GatherV2Tensordot_1/Shape_1Tensordot_1/axes_1Tensordot_1/GatherV2_3/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
]
Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_1/Prod_2ProdTensordot_1/GatherV2_2Tensordot_1/Const_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_1/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_1/Prod_3ProdTensordot_1/GatherV2_3Tensordot_1/Const_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
[
Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Э
Tensordot_1/concat_1ConcatV2Tensordot_1/axes_1Tensordot_1/free_1Tensordot_1/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
}
Tensordot_1/stack_1PackTensordot_1/Prod_3Tensordot_1/Prod_2*
N*
T0*
_output_shapes
:*

axis 
Е
Tensordot_1/transpose_1	Transposel2_normalize_3Tensordot_1/concat_1*
T0*
Tperm0*#
_output_shapes
:         
Ч
Tensordot_1/Reshape_1ReshapeTensordot_1/transpose_1Tensordot_1/stack_1*
T0*
Tshape0*0
_output_shapes
:                  
й
Tensordot_1/MatMulMatMulTensordot_1/ReshapeTensordot_1/Reshape_1*
T0*0
_output_shapes
:                  *
transpose_a( *
transpose_b( 
[
Tensordot_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
б
Tensordot_1/concat_2ConcatV2Tensordot_1/GatherV2Tensordot_1/GatherV2_2Tensordot_1/concat_2/axis*
N*
T0*

Tidx0*
_output_shapes
: 
o
Tensordot_1ReshapeTensordot_1/MatMulTensordot_1/concat_2*
T0*
Tshape0*
_output_shapes
: 
`
Identity_10Identityconv2d/Conv2D*
T0*/
_output_shapes
:         
c
Reshape_47/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
r

Reshape_47Reshapeconv2d/Conv2DReshape_47/shape*
T0*
Tshape0*#
_output_shapes
:         
Y
l2_normalize_4/SquareSquare
Reshape_47*
T0*#
_output_shapes
:         
^
l2_normalize_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
И
l2_normalize_4/SumSuml2_normalize_4/Squarel2_normalize_4/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
]
l2_normalize_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
t
l2_normalize_4/MaximumMaximuml2_normalize_4/Suml2_normalize_4/Maximum/y*
T0*
_output_shapes
:
Z
l2_normalize_4/RsqrtRsqrtl2_normalize_4/Maximum*
T0*
_output_shapes
:
e
l2_normalize_4Mul
Reshape_47l2_normalize_4/Rsqrt*
T0*#
_output_shapes
:         
c
Reshape_48/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
p

Reshape_48ReshapeIdentity_10Reshape_48/shape*
T0*
Tshape0*#
_output_shapes
:         
Y
l2_normalize_5/SquareSquare
Reshape_48*
T0*#
_output_shapes
:         
^
l2_normalize_5/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
И
l2_normalize_5/SumSuml2_normalize_5/Squarel2_normalize_5/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
]
l2_normalize_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
t
l2_normalize_5/MaximumMaximuml2_normalize_5/Suml2_normalize_5/Maximum/y*
T0*
_output_shapes
:
Z
l2_normalize_5/RsqrtRsqrtl2_normalize_5/Maximum*
T0*
_output_shapes
:
e
l2_normalize_5Mul
Reshape_48l2_normalize_5/Rsqrt*
T0*#
_output_shapes
:         
Z
Tensordot_2/axesConst*
_output_shapes
:*
dtype0*
valueB: 
S
Tensordot_2/freeConst*
_output_shapes
: *
dtype0*
valueB 
_
Tensordot_2/ShapeShapel2_normalize_4*
T0*
_output_shapes
:*
out_type0
[
Tensordot_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╕
Tensordot_2/GatherV2GatherV2Tensordot_2/ShapeTensordot_2/freeTensordot_2/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╛
Tensordot_2/GatherV2_1GatherV2Tensordot_2/ShapeTensordot_2/axesTensordot_2/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
[
Tensordot_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

Tensordot_2/ProdProdTensordot_2/GatherV2Tensordot_2/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_2/Prod_1ProdTensordot_2/GatherV2_1Tensordot_2/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Y
Tensordot_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Х
Tensordot_2/concatConcatV2Tensordot_2/freeTensordot_2/axesTensordot_2/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
y
Tensordot_2/stackPackTensordot_2/ProdTensordot_2/Prod_1*
N*
T0*
_output_shapes
:*

axis 
Б
Tensordot_2/transpose	Transposel2_normalize_4Tensordot_2/concat*
T0*
Tperm0*#
_output_shapes
:         
С
Tensordot_2/ReshapeReshapeTensordot_2/transposeTensordot_2/stack*
T0*
Tshape0*0
_output_shapes
:                  
\
Tensordot_2/axes_1Const*
_output_shapes
:*
dtype0*
valueB: 
U
Tensordot_2/free_1Const*
_output_shapes
: *
dtype0*
valueB 
a
Tensordot_2/Shape_1Shapel2_normalize_5*
T0*
_output_shapes
:*
out_type0
]
Tensordot_2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
└
Tensordot_2/GatherV2_2GatherV2Tensordot_2/Shape_1Tensordot_2/free_1Tensordot_2/GatherV2_2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_2/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
┬
Tensordot_2/GatherV2_3GatherV2Tensordot_2/Shape_1Tensordot_2/axes_1Tensordot_2/GatherV2_3/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
]
Tensordot_2/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_2/Prod_2ProdTensordot_2/GatherV2_2Tensordot_2/Const_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_2/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_2/Prod_3ProdTensordot_2/GatherV2_3Tensordot_2/Const_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
[
Tensordot_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Э
Tensordot_2/concat_1ConcatV2Tensordot_2/axes_1Tensordot_2/free_1Tensordot_2/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
}
Tensordot_2/stack_1PackTensordot_2/Prod_3Tensordot_2/Prod_2*
N*
T0*
_output_shapes
:*

axis 
Е
Tensordot_2/transpose_1	Transposel2_normalize_5Tensordot_2/concat_1*
T0*
Tperm0*#
_output_shapes
:         
Ч
Tensordot_2/Reshape_1ReshapeTensordot_2/transpose_1Tensordot_2/stack_1*
T0*
Tshape0*0
_output_shapes
:                  
й
Tensordot_2/MatMulMatMulTensordot_2/ReshapeTensordot_2/Reshape_1*
T0*0
_output_shapes
:                  *
transpose_a( *
transpose_b( 
[
Tensordot_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
б
Tensordot_2/concat_2ConcatV2Tensordot_2/GatherV2Tensordot_2/GatherV2_2Tensordot_2/concat_2/axis*
N*
T0*

Tidx0*
_output_shapes
: 
o
Tensordot_2ReshapeTensordot_2/MatMulTensordot_2/concat_2*
T0*
Tshape0*
_output_shapes
: 
`
Identity_11Identityconv2d/Conv2D*
T0*/
_output_shapes
:         
c
Reshape_49/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
r

Reshape_49Reshapeconv2d/Conv2DReshape_49/shape*
T0*
Tshape0*#
_output_shapes
:         
Y
l2_normalize_6/SquareSquare
Reshape_49*
T0*#
_output_shapes
:         
^
l2_normalize_6/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
И
l2_normalize_6/SumSuml2_normalize_6/Squarel2_normalize_6/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
]
l2_normalize_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
t
l2_normalize_6/MaximumMaximuml2_normalize_6/Suml2_normalize_6/Maximum/y*
T0*
_output_shapes
:
Z
l2_normalize_6/RsqrtRsqrtl2_normalize_6/Maximum*
T0*
_output_shapes
:
e
l2_normalize_6Mul
Reshape_49l2_normalize_6/Rsqrt*
T0*#
_output_shapes
:         
c
Reshape_50/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
p

Reshape_50ReshapeIdentity_11Reshape_50/shape*
T0*
Tshape0*#
_output_shapes
:         
Y
l2_normalize_7/SquareSquare
Reshape_50*
T0*#
_output_shapes
:         
^
l2_normalize_7/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
И
l2_normalize_7/SumSuml2_normalize_7/Squarel2_normalize_7/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
]
l2_normalize_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
t
l2_normalize_7/MaximumMaximuml2_normalize_7/Suml2_normalize_7/Maximum/y*
T0*
_output_shapes
:
Z
l2_normalize_7/RsqrtRsqrtl2_normalize_7/Maximum*
T0*
_output_shapes
:
e
l2_normalize_7Mul
Reshape_50l2_normalize_7/Rsqrt*
T0*#
_output_shapes
:         
Z
Tensordot_3/axesConst*
_output_shapes
:*
dtype0*
valueB: 
S
Tensordot_3/freeConst*
_output_shapes
: *
dtype0*
valueB 
_
Tensordot_3/ShapeShapel2_normalize_6*
T0*
_output_shapes
:*
out_type0
[
Tensordot_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╕
Tensordot_3/GatherV2GatherV2Tensordot_3/ShapeTensordot_3/freeTensordot_3/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╛
Tensordot_3/GatherV2_1GatherV2Tensordot_3/ShapeTensordot_3/axesTensordot_3/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
[
Tensordot_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

Tensordot_3/ProdProdTensordot_3/GatherV2Tensordot_3/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_3/Prod_1ProdTensordot_3/GatherV2_1Tensordot_3/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Y
Tensordot_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Х
Tensordot_3/concatConcatV2Tensordot_3/freeTensordot_3/axesTensordot_3/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
y
Tensordot_3/stackPackTensordot_3/ProdTensordot_3/Prod_1*
N*
T0*
_output_shapes
:*

axis 
Б
Tensordot_3/transpose	Transposel2_normalize_6Tensordot_3/concat*
T0*
Tperm0*#
_output_shapes
:         
С
Tensordot_3/ReshapeReshapeTensordot_3/transposeTensordot_3/stack*
T0*
Tshape0*0
_output_shapes
:                  
\
Tensordot_3/axes_1Const*
_output_shapes
:*
dtype0*
valueB: 
U
Tensordot_3/free_1Const*
_output_shapes
: *
dtype0*
valueB 
a
Tensordot_3/Shape_1Shapel2_normalize_7*
T0*
_output_shapes
:*
out_type0
]
Tensordot_3/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
└
Tensordot_3/GatherV2_2GatherV2Tensordot_3/Shape_1Tensordot_3/free_1Tensordot_3/GatherV2_2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_3/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
┬
Tensordot_3/GatherV2_3GatherV2Tensordot_3/Shape_1Tensordot_3/axes_1Tensordot_3/GatherV2_3/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
]
Tensordot_3/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_3/Prod_2ProdTensordot_3/GatherV2_2Tensordot_3/Const_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_3/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_3/Prod_3ProdTensordot_3/GatherV2_3Tensordot_3/Const_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
[
Tensordot_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Э
Tensordot_3/concat_1ConcatV2Tensordot_3/axes_1Tensordot_3/free_1Tensordot_3/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
}
Tensordot_3/stack_1PackTensordot_3/Prod_3Tensordot_3/Prod_2*
N*
T0*
_output_shapes
:*

axis 
Е
Tensordot_3/transpose_1	Transposel2_normalize_7Tensordot_3/concat_1*
T0*
Tperm0*#
_output_shapes
:         
Ч
Tensordot_3/Reshape_1ReshapeTensordot_3/transpose_1Tensordot_3/stack_1*
T0*
Tshape0*0
_output_shapes
:                  
й
Tensordot_3/MatMulMatMulTensordot_3/ReshapeTensordot_3/Reshape_1*
T0*0
_output_shapes
:                  *
transpose_a( *
transpose_b( 
[
Tensordot_3/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
б
Tensordot_3/concat_2ConcatV2Tensordot_3/GatherV2Tensordot_3/GatherV2_2Tensordot_3/concat_2/axis*
N*
T0*

Tidx0*
_output_shapes
: 
o
Tensordot_3ReshapeTensordot_3/MatMulTensordot_3/concat_2*
T0*
Tshape0*
_output_shapes
: 
b
Identity_12Identityconv2d_1/Conv2D*
T0*/
_output_shapes
:         

c
Reshape_51/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
t

Reshape_51Reshapeconv2d_1/Conv2DReshape_51/shape*
T0*
Tshape0*#
_output_shapes
:         
Y
l2_normalize_8/SquareSquare
Reshape_51*
T0*#
_output_shapes
:         
^
l2_normalize_8/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
И
l2_normalize_8/SumSuml2_normalize_8/Squarel2_normalize_8/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
]
l2_normalize_8/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
t
l2_normalize_8/MaximumMaximuml2_normalize_8/Suml2_normalize_8/Maximum/y*
T0*
_output_shapes
:
Z
l2_normalize_8/RsqrtRsqrtl2_normalize_8/Maximum*
T0*
_output_shapes
:
e
l2_normalize_8Mul
Reshape_51l2_normalize_8/Rsqrt*
T0*#
_output_shapes
:         
c
Reshape_52/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
p

Reshape_52ReshapeIdentity_12Reshape_52/shape*
T0*
Tshape0*#
_output_shapes
:         
Y
l2_normalize_9/SquareSquare
Reshape_52*
T0*#
_output_shapes
:         
^
l2_normalize_9/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
И
l2_normalize_9/SumSuml2_normalize_9/Squarel2_normalize_9/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
]
l2_normalize_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
t
l2_normalize_9/MaximumMaximuml2_normalize_9/Suml2_normalize_9/Maximum/y*
T0*
_output_shapes
:
Z
l2_normalize_9/RsqrtRsqrtl2_normalize_9/Maximum*
T0*
_output_shapes
:
e
l2_normalize_9Mul
Reshape_52l2_normalize_9/Rsqrt*
T0*#
_output_shapes
:         
Z
Tensordot_4/axesConst*
_output_shapes
:*
dtype0*
valueB: 
S
Tensordot_4/freeConst*
_output_shapes
: *
dtype0*
valueB 
_
Tensordot_4/ShapeShapel2_normalize_8*
T0*
_output_shapes
:*
out_type0
[
Tensordot_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╕
Tensordot_4/GatherV2GatherV2Tensordot_4/ShapeTensordot_4/freeTensordot_4/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╛
Tensordot_4/GatherV2_1GatherV2Tensordot_4/ShapeTensordot_4/axesTensordot_4/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
[
Tensordot_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

Tensordot_4/ProdProdTensordot_4/GatherV2Tensordot_4/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_4/Prod_1ProdTensordot_4/GatherV2_1Tensordot_4/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Y
Tensordot_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Х
Tensordot_4/concatConcatV2Tensordot_4/freeTensordot_4/axesTensordot_4/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
y
Tensordot_4/stackPackTensordot_4/ProdTensordot_4/Prod_1*
N*
T0*
_output_shapes
:*

axis 
Б
Tensordot_4/transpose	Transposel2_normalize_8Tensordot_4/concat*
T0*
Tperm0*#
_output_shapes
:         
С
Tensordot_4/ReshapeReshapeTensordot_4/transposeTensordot_4/stack*
T0*
Tshape0*0
_output_shapes
:                  
\
Tensordot_4/axes_1Const*
_output_shapes
:*
dtype0*
valueB: 
U
Tensordot_4/free_1Const*
_output_shapes
: *
dtype0*
valueB 
a
Tensordot_4/Shape_1Shapel2_normalize_9*
T0*
_output_shapes
:*
out_type0
]
Tensordot_4/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
└
Tensordot_4/GatherV2_2GatherV2Tensordot_4/Shape_1Tensordot_4/free_1Tensordot_4/GatherV2_2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_4/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
┬
Tensordot_4/GatherV2_3GatherV2Tensordot_4/Shape_1Tensordot_4/axes_1Tensordot_4/GatherV2_3/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
]
Tensordot_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_4/Prod_2ProdTensordot_4/GatherV2_2Tensordot_4/Const_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_4/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_4/Prod_3ProdTensordot_4/GatherV2_3Tensordot_4/Const_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
[
Tensordot_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Э
Tensordot_4/concat_1ConcatV2Tensordot_4/axes_1Tensordot_4/free_1Tensordot_4/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
}
Tensordot_4/stack_1PackTensordot_4/Prod_3Tensordot_4/Prod_2*
N*
T0*
_output_shapes
:*

axis 
Е
Tensordot_4/transpose_1	Transposel2_normalize_9Tensordot_4/concat_1*
T0*
Tperm0*#
_output_shapes
:         
Ч
Tensordot_4/Reshape_1ReshapeTensordot_4/transpose_1Tensordot_4/stack_1*
T0*
Tshape0*0
_output_shapes
:                  
й
Tensordot_4/MatMulMatMulTensordot_4/ReshapeTensordot_4/Reshape_1*
T0*0
_output_shapes
:                  *
transpose_a( *
transpose_b( 
[
Tensordot_4/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
б
Tensordot_4/concat_2ConcatV2Tensordot_4/GatherV2Tensordot_4/GatherV2_2Tensordot_4/concat_2/axis*
N*
T0*

Tidx0*
_output_shapes
: 
o
Tensordot_4ReshapeTensordot_4/MatMulTensordot_4/concat_2*
T0*
Tshape0*
_output_shapes
: 
b
Identity_13Identityconv2d_1/Conv2D*
T0*/
_output_shapes
:         

c
Reshape_53/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
t

Reshape_53Reshapeconv2d_1/Conv2DReshape_53/shape*
T0*
Tshape0*#
_output_shapes
:         
Z
l2_normalize_10/SquareSquare
Reshape_53*
T0*#
_output_shapes
:         
_
l2_normalize_10/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_10/SumSuml2_normalize_10/Squarel2_normalize_10/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_10/MaximumMaximuml2_normalize_10/Suml2_normalize_10/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_10/RsqrtRsqrtl2_normalize_10/Maximum*
T0*
_output_shapes
:
g
l2_normalize_10Mul
Reshape_53l2_normalize_10/Rsqrt*
T0*#
_output_shapes
:         
c
Reshape_54/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
p

Reshape_54ReshapeIdentity_13Reshape_54/shape*
T0*
Tshape0*#
_output_shapes
:         
Z
l2_normalize_11/SquareSquare
Reshape_54*
T0*#
_output_shapes
:         
_
l2_normalize_11/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_11/SumSuml2_normalize_11/Squarel2_normalize_11/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_11/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_11/MaximumMaximuml2_normalize_11/Suml2_normalize_11/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_11/RsqrtRsqrtl2_normalize_11/Maximum*
T0*
_output_shapes
:
g
l2_normalize_11Mul
Reshape_54l2_normalize_11/Rsqrt*
T0*#
_output_shapes
:         
Z
Tensordot_5/axesConst*
_output_shapes
:*
dtype0*
valueB: 
S
Tensordot_5/freeConst*
_output_shapes
: *
dtype0*
valueB 
`
Tensordot_5/ShapeShapel2_normalize_10*
T0*
_output_shapes
:*
out_type0
[
Tensordot_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╕
Tensordot_5/GatherV2GatherV2Tensordot_5/ShapeTensordot_5/freeTensordot_5/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_5/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╛
Tensordot_5/GatherV2_1GatherV2Tensordot_5/ShapeTensordot_5/axesTensordot_5/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
[
Tensordot_5/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

Tensordot_5/ProdProdTensordot_5/GatherV2Tensordot_5/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_5/Prod_1ProdTensordot_5/GatherV2_1Tensordot_5/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Y
Tensordot_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Х
Tensordot_5/concatConcatV2Tensordot_5/freeTensordot_5/axesTensordot_5/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
y
Tensordot_5/stackPackTensordot_5/ProdTensordot_5/Prod_1*
N*
T0*
_output_shapes
:*

axis 
В
Tensordot_5/transpose	Transposel2_normalize_10Tensordot_5/concat*
T0*
Tperm0*#
_output_shapes
:         
С
Tensordot_5/ReshapeReshapeTensordot_5/transposeTensordot_5/stack*
T0*
Tshape0*0
_output_shapes
:                  
\
Tensordot_5/axes_1Const*
_output_shapes
:*
dtype0*
valueB: 
U
Tensordot_5/free_1Const*
_output_shapes
: *
dtype0*
valueB 
b
Tensordot_5/Shape_1Shapel2_normalize_11*
T0*
_output_shapes
:*
out_type0
]
Tensordot_5/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
└
Tensordot_5/GatherV2_2GatherV2Tensordot_5/Shape_1Tensordot_5/free_1Tensordot_5/GatherV2_2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_5/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
┬
Tensordot_5/GatherV2_3GatherV2Tensordot_5/Shape_1Tensordot_5/axes_1Tensordot_5/GatherV2_3/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
]
Tensordot_5/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_5/Prod_2ProdTensordot_5/GatherV2_2Tensordot_5/Const_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_5/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_5/Prod_3ProdTensordot_5/GatherV2_3Tensordot_5/Const_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
[
Tensordot_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Э
Tensordot_5/concat_1ConcatV2Tensordot_5/axes_1Tensordot_5/free_1Tensordot_5/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
}
Tensordot_5/stack_1PackTensordot_5/Prod_3Tensordot_5/Prod_2*
N*
T0*
_output_shapes
:*

axis 
Ж
Tensordot_5/transpose_1	Transposel2_normalize_11Tensordot_5/concat_1*
T0*
Tperm0*#
_output_shapes
:         
Ч
Tensordot_5/Reshape_1ReshapeTensordot_5/transpose_1Tensordot_5/stack_1*
T0*
Tshape0*0
_output_shapes
:                  
й
Tensordot_5/MatMulMatMulTensordot_5/ReshapeTensordot_5/Reshape_1*
T0*0
_output_shapes
:                  *
transpose_a( *
transpose_b( 
[
Tensordot_5/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
б
Tensordot_5/concat_2ConcatV2Tensordot_5/GatherV2Tensordot_5/GatherV2_2Tensordot_5/concat_2/axis*
N*
T0*

Tidx0*
_output_shapes
: 
o
Tensordot_5ReshapeTensordot_5/MatMulTensordot_5/concat_2*
T0*
Tshape0*
_output_shapes
: 
Y
Abs_48Absconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
N
	add_133/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
^
add_133AddV2Abs_48	add_133/y*
T0*0
_output_shapes
:         А
Q
Log_96Logadd_133*
T0*0
_output_shapes
:         А
M
Const_79Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_97LogConst_79*
T0*
_output_shapes
: 
`

truediv_48RealDivLog_96Log_97*
T0*0
_output_shapes
:         А
Q
ReadVariableOp_72ReadVariableOpslope_3*
_output_shapes
: *
dtype0
g
mul_63MulReadVariableOp_72
truediv_48*
T0*0
_output_shapes
:         А
U
ReadVariableOp_73ReadVariableOpintercept_3*
_output_shapes
: *
dtype0
f
add_134AddV2ReadVariableOp_73mul_63*
T0*0
_output_shapes
:         А
Е
differentiable_round_36Roundadd_134*
T0*
_gradient_op_type
Identity*0
_output_shapes
:         А
V
GreaterEqual_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
~
GreaterEqual_12GreaterEqualconv2d_2/Conv2DGreaterEqual_12/y*
T0*0
_output_shapes
:         А
a
ones_like_24/ShapeShapeconv2d_2/Conv2D*
T0*
_output_shapes
:*
out_type0
W
ones_like_24/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Й
ones_like_24Fillones_like_24/Shapeones_like_24/Const*
T0*0
_output_shapes
:         А*

index_type0
f
zeros_like_24	ZerosLikeconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
А
SelectV2_24SelectV2GreaterEqual_12ones_like_24zeros_like_24*
T0*0
_output_shapes
:         А
S
LessEqual_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
u
LessEqual_12	LessEqualconv2d_2/Conv2DLessEqual_12/y*
T0*0
_output_shapes
:         А
a
ones_like_25/ShapeShapeconv2d_2/Conv2D*
T0*
_output_shapes
:*
out_type0
W
ones_like_25/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Й
ones_like_25Fillones_like_25/Shapeones_like_25/Const*
T0*0
_output_shapes
:         А*

index_type0
M
mul_64/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
`
mul_64Mulmul_64/xones_like_25*
T0*0
_output_shapes
:         А
f
zeros_like_25	ZerosLikeconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
w
SelectV2_25SelectV2LessEqual_12mul_64zeros_like_25*
T0*0
_output_shapes
:         А
c
Add_135AddSelectV2_25SelectV2_24*
T0*0
_output_shapes
:         А
M
pow_18/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
k
pow_18Powpow_18/xdifferentiable_round_36*
T0*0
_output_shapes
:         А
Y
mul_65MulAdd_135pow_18*
T0*0
_output_shapes
:         А
c
Reshape_55/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
t

Reshape_55Reshapeconv2d_2/Conv2DReshape_55/shape*
T0*
Tshape0*#
_output_shapes
:         
Z
l2_normalize_12/SquareSquare
Reshape_55*
T0*#
_output_shapes
:         
_
l2_normalize_12/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_12/SumSuml2_normalize_12/Squarel2_normalize_12/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_12/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_12/MaximumMaximuml2_normalize_12/Suml2_normalize_12/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_12/RsqrtRsqrtl2_normalize_12/Maximum*
T0*
_output_shapes
:
g
l2_normalize_12Mul
Reshape_55l2_normalize_12/Rsqrt*
T0*#
_output_shapes
:         
c
Reshape_56/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
k

Reshape_56Reshapemul_65Reshape_56/shape*
T0*
Tshape0*#
_output_shapes
:         
Z
l2_normalize_13/SquareSquare
Reshape_56*
T0*#
_output_shapes
:         
_
l2_normalize_13/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_13/SumSuml2_normalize_13/Squarel2_normalize_13/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_13/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_13/MaximumMaximuml2_normalize_13/Suml2_normalize_13/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_13/RsqrtRsqrtl2_normalize_13/Maximum*
T0*
_output_shapes
:
g
l2_normalize_13Mul
Reshape_56l2_normalize_13/Rsqrt*
T0*#
_output_shapes
:         
Z
Tensordot_6/axesConst*
_output_shapes
:*
dtype0*
valueB: 
S
Tensordot_6/freeConst*
_output_shapes
: *
dtype0*
valueB 
`
Tensordot_6/ShapeShapel2_normalize_12*
T0*
_output_shapes
:*
out_type0
[
Tensordot_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╕
Tensordot_6/GatherV2GatherV2Tensordot_6/ShapeTensordot_6/freeTensordot_6/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_6/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╛
Tensordot_6/GatherV2_1GatherV2Tensordot_6/ShapeTensordot_6/axesTensordot_6/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
[
Tensordot_6/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

Tensordot_6/ProdProdTensordot_6/GatherV2Tensordot_6/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_6/Prod_1ProdTensordot_6/GatherV2_1Tensordot_6/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Y
Tensordot_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Х
Tensordot_6/concatConcatV2Tensordot_6/freeTensordot_6/axesTensordot_6/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
y
Tensordot_6/stackPackTensordot_6/ProdTensordot_6/Prod_1*
N*
T0*
_output_shapes
:*

axis 
В
Tensordot_6/transpose	Transposel2_normalize_12Tensordot_6/concat*
T0*
Tperm0*#
_output_shapes
:         
С
Tensordot_6/ReshapeReshapeTensordot_6/transposeTensordot_6/stack*
T0*
Tshape0*0
_output_shapes
:                  
\
Tensordot_6/axes_1Const*
_output_shapes
:*
dtype0*
valueB: 
U
Tensordot_6/free_1Const*
_output_shapes
: *
dtype0*
valueB 
b
Tensordot_6/Shape_1Shapel2_normalize_13*
T0*
_output_shapes
:*
out_type0
]
Tensordot_6/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
└
Tensordot_6/GatherV2_2GatherV2Tensordot_6/Shape_1Tensordot_6/free_1Tensordot_6/GatherV2_2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_6/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
┬
Tensordot_6/GatherV2_3GatherV2Tensordot_6/Shape_1Tensordot_6/axes_1Tensordot_6/GatherV2_3/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
]
Tensordot_6/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_6/Prod_2ProdTensordot_6/GatherV2_2Tensordot_6/Const_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_6/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_6/Prod_3ProdTensordot_6/GatherV2_3Tensordot_6/Const_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
[
Tensordot_6/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Э
Tensordot_6/concat_1ConcatV2Tensordot_6/axes_1Tensordot_6/free_1Tensordot_6/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
}
Tensordot_6/stack_1PackTensordot_6/Prod_3Tensordot_6/Prod_2*
N*
T0*
_output_shapes
:*

axis 
Ж
Tensordot_6/transpose_1	Transposel2_normalize_13Tensordot_6/concat_1*
T0*
Tperm0*#
_output_shapes
:         
Ч
Tensordot_6/Reshape_1ReshapeTensordot_6/transpose_1Tensordot_6/stack_1*
T0*
Tshape0*0
_output_shapes
:                  
й
Tensordot_6/MatMulMatMulTensordot_6/ReshapeTensordot_6/Reshape_1*
T0*0
_output_shapes
:                  *
transpose_a( *
transpose_b( 
[
Tensordot_6/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
б
Tensordot_6/concat_2ConcatV2Tensordot_6/GatherV2Tensordot_6/GatherV2_2Tensordot_6/concat_2/axis*
N*
T0*

Tidx0*
_output_shapes
: 
o
Tensordot_6ReshapeTensordot_6/MatMulTensordot_6/concat_2*
T0*
Tshape0*
_output_shapes
: 
Y
Abs_49Absconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
N
	add_136/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
^
add_136AddV2Abs_49	add_136/y*
T0*0
_output_shapes
:         А
Q
Log_98Logadd_136*
T0*0
_output_shapes
:         А
M
Const_80Const*
_output_shapes
: *
dtype0*
valueB
 *  @
8
Log_99LogConst_80*
T0*
_output_shapes
: 
`

truediv_49RealDivLog_98Log_99*
T0*0
_output_shapes
:         А
Q
ReadVariableOp_74ReadVariableOpslope_3*
_output_shapes
: *
dtype0
g
mul_66MulReadVariableOp_74
truediv_49*
T0*0
_output_shapes
:         А
U
ReadVariableOp_75ReadVariableOpintercept_3*
_output_shapes
: *
dtype0
f
add_137AddV2ReadVariableOp_75mul_66*
T0*0
_output_shapes
:         А
Е
differentiable_round_37Roundadd_137*
T0*
_gradient_op_type
Identity*0
_output_shapes
:         А
V
GreaterEqual_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
~
GreaterEqual_13GreaterEqualconv2d_2/Conv2DGreaterEqual_13/y*
T0*0
_output_shapes
:         А
a
ones_like_26/ShapeShapeconv2d_2/Conv2D*
T0*
_output_shapes
:*
out_type0
W
ones_like_26/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Й
ones_like_26Fillones_like_26/Shapeones_like_26/Const*
T0*0
_output_shapes
:         А*

index_type0
f
zeros_like_26	ZerosLikeconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
А
SelectV2_26SelectV2GreaterEqual_13ones_like_26zeros_like_26*
T0*0
_output_shapes
:         А
S
LessEqual_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
u
LessEqual_13	LessEqualconv2d_2/Conv2DLessEqual_13/y*
T0*0
_output_shapes
:         А
a
ones_like_27/ShapeShapeconv2d_2/Conv2D*
T0*
_output_shapes
:*
out_type0
W
ones_like_27/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Й
ones_like_27Fillones_like_27/Shapeones_like_27/Const*
T0*0
_output_shapes
:         А*

index_type0
M
mul_67/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
`
mul_67Mulmul_67/xones_like_27*
T0*0
_output_shapes
:         А
f
zeros_like_27	ZerosLikeconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
w
SelectV2_27SelectV2LessEqual_13mul_67zeros_like_27*
T0*0
_output_shapes
:         А
c
Add_138AddSelectV2_27SelectV2_26*
T0*0
_output_shapes
:         А
M
pow_19/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
k
pow_19Powpow_19/xdifferentiable_round_37*
T0*0
_output_shapes
:         А
Y
mul_68MulAdd_138pow_19*
T0*0
_output_shapes
:         А
c
Reshape_57/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
t

Reshape_57Reshapeconv2d_2/Conv2DReshape_57/shape*
T0*
Tshape0*#
_output_shapes
:         
Z
l2_normalize_14/SquareSquare
Reshape_57*
T0*#
_output_shapes
:         
_
l2_normalize_14/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_14/SumSuml2_normalize_14/Squarel2_normalize_14/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_14/MaximumMaximuml2_normalize_14/Suml2_normalize_14/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_14/RsqrtRsqrtl2_normalize_14/Maximum*
T0*
_output_shapes
:
g
l2_normalize_14Mul
Reshape_57l2_normalize_14/Rsqrt*
T0*#
_output_shapes
:         
c
Reshape_58/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
k

Reshape_58Reshapemul_68Reshape_58/shape*
T0*
Tshape0*#
_output_shapes
:         
Z
l2_normalize_15/SquareSquare
Reshape_58*
T0*#
_output_shapes
:         
_
l2_normalize_15/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_15/SumSuml2_normalize_15/Squarel2_normalize_15/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_15/MaximumMaximuml2_normalize_15/Suml2_normalize_15/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_15/RsqrtRsqrtl2_normalize_15/Maximum*
T0*
_output_shapes
:
g
l2_normalize_15Mul
Reshape_58l2_normalize_15/Rsqrt*
T0*#
_output_shapes
:         
Z
Tensordot_7/axesConst*
_output_shapes
:*
dtype0*
valueB: 
S
Tensordot_7/freeConst*
_output_shapes
: *
dtype0*
valueB 
`
Tensordot_7/ShapeShapel2_normalize_14*
T0*
_output_shapes
:*
out_type0
[
Tensordot_7/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╕
Tensordot_7/GatherV2GatherV2Tensordot_7/ShapeTensordot_7/freeTensordot_7/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_7/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╛
Tensordot_7/GatherV2_1GatherV2Tensordot_7/ShapeTensordot_7/axesTensordot_7/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
[
Tensordot_7/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

Tensordot_7/ProdProdTensordot_7/GatherV2Tensordot_7/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_7/Prod_1ProdTensordot_7/GatherV2_1Tensordot_7/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Y
Tensordot_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Х
Tensordot_7/concatConcatV2Tensordot_7/freeTensordot_7/axesTensordot_7/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
y
Tensordot_7/stackPackTensordot_7/ProdTensordot_7/Prod_1*
N*
T0*
_output_shapes
:*

axis 
В
Tensordot_7/transpose	Transposel2_normalize_14Tensordot_7/concat*
T0*
Tperm0*#
_output_shapes
:         
С
Tensordot_7/ReshapeReshapeTensordot_7/transposeTensordot_7/stack*
T0*
Tshape0*0
_output_shapes
:                  
\
Tensordot_7/axes_1Const*
_output_shapes
:*
dtype0*
valueB: 
U
Tensordot_7/free_1Const*
_output_shapes
: *
dtype0*
valueB 
b
Tensordot_7/Shape_1Shapel2_normalize_15*
T0*
_output_shapes
:*
out_type0
]
Tensordot_7/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
└
Tensordot_7/GatherV2_2GatherV2Tensordot_7/Shape_1Tensordot_7/free_1Tensordot_7/GatherV2_2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_7/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
┬
Tensordot_7/GatherV2_3GatherV2Tensordot_7/Shape_1Tensordot_7/axes_1Tensordot_7/GatherV2_3/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
]
Tensordot_7/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_7/Prod_2ProdTensordot_7/GatherV2_2Tensordot_7/Const_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_7/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_7/Prod_3ProdTensordot_7/GatherV2_3Tensordot_7/Const_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
[
Tensordot_7/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Э
Tensordot_7/concat_1ConcatV2Tensordot_7/axes_1Tensordot_7/free_1Tensordot_7/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
}
Tensordot_7/stack_1PackTensordot_7/Prod_3Tensordot_7/Prod_2*
N*
T0*
_output_shapes
:*

axis 
Ж
Tensordot_7/transpose_1	Transposel2_normalize_15Tensordot_7/concat_1*
T0*
Tperm0*#
_output_shapes
:         
Ч
Tensordot_7/Reshape_1ReshapeTensordot_7/transpose_1Tensordot_7/stack_1*
T0*
Tshape0*0
_output_shapes
:                  
й
Tensordot_7/MatMulMatMulTensordot_7/ReshapeTensordot_7/Reshape_1*
T0*0
_output_shapes
:                  *
transpose_a( *
transpose_b( 
[
Tensordot_7/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
б
Tensordot_7/concat_2ConcatV2Tensordot_7/GatherV2Tensordot_7/GatherV2_2Tensordot_7/concat_2/axis*
N*
T0*

Tidx0*
_output_shapes
: 
o
Tensordot_7ReshapeTensordot_7/MatMulTensordot_7/concat_2*
T0*
Tshape0*
_output_shapes
: 
c
Identity_14Identityconv2d_3/Conv2D*
T0*0
_output_shapes
:         А
c
Reshape_59/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
t

Reshape_59Reshapeconv2d_3/Conv2DReshape_59/shape*
T0*
Tshape0*#
_output_shapes
:         
Z
l2_normalize_16/SquareSquare
Reshape_59*
T0*#
_output_shapes
:         
_
l2_normalize_16/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_16/SumSuml2_normalize_16/Squarel2_normalize_16/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_16/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_16/MaximumMaximuml2_normalize_16/Suml2_normalize_16/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_16/RsqrtRsqrtl2_normalize_16/Maximum*
T0*
_output_shapes
:
g
l2_normalize_16Mul
Reshape_59l2_normalize_16/Rsqrt*
T0*#
_output_shapes
:         
c
Reshape_60/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
p

Reshape_60ReshapeIdentity_14Reshape_60/shape*
T0*
Tshape0*#
_output_shapes
:         
Z
l2_normalize_17/SquareSquare
Reshape_60*
T0*#
_output_shapes
:         
_
l2_normalize_17/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_17/SumSuml2_normalize_17/Squarel2_normalize_17/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_17/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_17/MaximumMaximuml2_normalize_17/Suml2_normalize_17/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_17/RsqrtRsqrtl2_normalize_17/Maximum*
T0*
_output_shapes
:
g
l2_normalize_17Mul
Reshape_60l2_normalize_17/Rsqrt*
T0*#
_output_shapes
:         
Z
Tensordot_8/axesConst*
_output_shapes
:*
dtype0*
valueB: 
S
Tensordot_8/freeConst*
_output_shapes
: *
dtype0*
valueB 
`
Tensordot_8/ShapeShapel2_normalize_16*
T0*
_output_shapes
:*
out_type0
[
Tensordot_8/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╕
Tensordot_8/GatherV2GatherV2Tensordot_8/ShapeTensordot_8/freeTensordot_8/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_8/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╛
Tensordot_8/GatherV2_1GatherV2Tensordot_8/ShapeTensordot_8/axesTensordot_8/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
[
Tensordot_8/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

Tensordot_8/ProdProdTensordot_8/GatherV2Tensordot_8/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_8/Prod_1ProdTensordot_8/GatherV2_1Tensordot_8/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Y
Tensordot_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Х
Tensordot_8/concatConcatV2Tensordot_8/freeTensordot_8/axesTensordot_8/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
y
Tensordot_8/stackPackTensordot_8/ProdTensordot_8/Prod_1*
N*
T0*
_output_shapes
:*

axis 
В
Tensordot_8/transpose	Transposel2_normalize_16Tensordot_8/concat*
T0*
Tperm0*#
_output_shapes
:         
С
Tensordot_8/ReshapeReshapeTensordot_8/transposeTensordot_8/stack*
T0*
Tshape0*0
_output_shapes
:                  
\
Tensordot_8/axes_1Const*
_output_shapes
:*
dtype0*
valueB: 
U
Tensordot_8/free_1Const*
_output_shapes
: *
dtype0*
valueB 
b
Tensordot_8/Shape_1Shapel2_normalize_17*
T0*
_output_shapes
:*
out_type0
]
Tensordot_8/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
└
Tensordot_8/GatherV2_2GatherV2Tensordot_8/Shape_1Tensordot_8/free_1Tensordot_8/GatherV2_2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_8/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
┬
Tensordot_8/GatherV2_3GatherV2Tensordot_8/Shape_1Tensordot_8/axes_1Tensordot_8/GatherV2_3/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
]
Tensordot_8/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_8/Prod_2ProdTensordot_8/GatherV2_2Tensordot_8/Const_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_8/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_8/Prod_3ProdTensordot_8/GatherV2_3Tensordot_8/Const_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
[
Tensordot_8/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Э
Tensordot_8/concat_1ConcatV2Tensordot_8/axes_1Tensordot_8/free_1Tensordot_8/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
}
Tensordot_8/stack_1PackTensordot_8/Prod_3Tensordot_8/Prod_2*
N*
T0*
_output_shapes
:*

axis 
Ж
Tensordot_8/transpose_1	Transposel2_normalize_17Tensordot_8/concat_1*
T0*
Tperm0*#
_output_shapes
:         
Ч
Tensordot_8/Reshape_1ReshapeTensordot_8/transpose_1Tensordot_8/stack_1*
T0*
Tshape0*0
_output_shapes
:                  
й
Tensordot_8/MatMulMatMulTensordot_8/ReshapeTensordot_8/Reshape_1*
T0*0
_output_shapes
:                  *
transpose_a( *
transpose_b( 
[
Tensordot_8/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
б
Tensordot_8/concat_2ConcatV2Tensordot_8/GatherV2Tensordot_8/GatherV2_2Tensordot_8/concat_2/axis*
N*
T0*

Tidx0*
_output_shapes
: 
o
Tensordot_8ReshapeTensordot_8/MatMulTensordot_8/concat_2*
T0*
Tshape0*
_output_shapes
: 
c
Identity_15Identityconv2d_3/Conv2D*
T0*0
_output_shapes
:         А
c
Reshape_61/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
t

Reshape_61Reshapeconv2d_3/Conv2DReshape_61/shape*
T0*
Tshape0*#
_output_shapes
:         
Z
l2_normalize_18/SquareSquare
Reshape_61*
T0*#
_output_shapes
:         
_
l2_normalize_18/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_18/SumSuml2_normalize_18/Squarel2_normalize_18/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_18/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_18/MaximumMaximuml2_normalize_18/Suml2_normalize_18/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_18/RsqrtRsqrtl2_normalize_18/Maximum*
T0*
_output_shapes
:
g
l2_normalize_18Mul
Reshape_61l2_normalize_18/Rsqrt*
T0*#
_output_shapes
:         
c
Reshape_62/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
p

Reshape_62ReshapeIdentity_15Reshape_62/shape*
T0*
Tshape0*#
_output_shapes
:         
Z
l2_normalize_19/SquareSquare
Reshape_62*
T0*#
_output_shapes
:         
_
l2_normalize_19/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_19/SumSuml2_normalize_19/Squarel2_normalize_19/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_19/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_19/MaximumMaximuml2_normalize_19/Suml2_normalize_19/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_19/RsqrtRsqrtl2_normalize_19/Maximum*
T0*
_output_shapes
:
g
l2_normalize_19Mul
Reshape_62l2_normalize_19/Rsqrt*
T0*#
_output_shapes
:         
Z
Tensordot_9/axesConst*
_output_shapes
:*
dtype0*
valueB: 
S
Tensordot_9/freeConst*
_output_shapes
: *
dtype0*
valueB 
`
Tensordot_9/ShapeShapel2_normalize_18*
T0*
_output_shapes
:*
out_type0
[
Tensordot_9/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╕
Tensordot_9/GatherV2GatherV2Tensordot_9/ShapeTensordot_9/freeTensordot_9/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_9/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╛
Tensordot_9/GatherV2_1GatherV2Tensordot_9/ShapeTensordot_9/axesTensordot_9/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
[
Tensordot_9/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

Tensordot_9/ProdProdTensordot_9/GatherV2Tensordot_9/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_9/Prod_1ProdTensordot_9/GatherV2_1Tensordot_9/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Y
Tensordot_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Х
Tensordot_9/concatConcatV2Tensordot_9/freeTensordot_9/axesTensordot_9/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
y
Tensordot_9/stackPackTensordot_9/ProdTensordot_9/Prod_1*
N*
T0*
_output_shapes
:*

axis 
В
Tensordot_9/transpose	Transposel2_normalize_18Tensordot_9/concat*
T0*
Tperm0*#
_output_shapes
:         
С
Tensordot_9/ReshapeReshapeTensordot_9/transposeTensordot_9/stack*
T0*
Tshape0*0
_output_shapes
:                  
\
Tensordot_9/axes_1Const*
_output_shapes
:*
dtype0*
valueB: 
U
Tensordot_9/free_1Const*
_output_shapes
: *
dtype0*
valueB 
b
Tensordot_9/Shape_1Shapel2_normalize_19*
T0*
_output_shapes
:*
out_type0
]
Tensordot_9/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
└
Tensordot_9/GatherV2_2GatherV2Tensordot_9/Shape_1Tensordot_9/free_1Tensordot_9/GatherV2_2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: *

batch_dims 
]
Tensordot_9/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
┬
Tensordot_9/GatherV2_3GatherV2Tensordot_9/Shape_1Tensordot_9/axes_1Tensordot_9/GatherV2_3/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
]
Tensordot_9/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_9/Prod_2ProdTensordot_9/GatherV2_2Tensordot_9/Const_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
]
Tensordot_9/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_9/Prod_3ProdTensordot_9/GatherV2_3Tensordot_9/Const_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
[
Tensordot_9/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Э
Tensordot_9/concat_1ConcatV2Tensordot_9/axes_1Tensordot_9/free_1Tensordot_9/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
}
Tensordot_9/stack_1PackTensordot_9/Prod_3Tensordot_9/Prod_2*
N*
T0*
_output_shapes
:*

axis 
Ж
Tensordot_9/transpose_1	Transposel2_normalize_19Tensordot_9/concat_1*
T0*
Tperm0*#
_output_shapes
:         
Ч
Tensordot_9/Reshape_1ReshapeTensordot_9/transpose_1Tensordot_9/stack_1*
T0*
Tshape0*0
_output_shapes
:                  
й
Tensordot_9/MatMulMatMulTensordot_9/ReshapeTensordot_9/Reshape_1*
T0*0
_output_shapes
:                  *
transpose_a( *
transpose_b( 
[
Tensordot_9/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
б
Tensordot_9/concat_2ConcatV2Tensordot_9/GatherV2Tensordot_9/GatherV2_2Tensordot_9/concat_2/axis*
N*
T0*

Tidx0*
_output_shapes
: 
o
Tensordot_9ReshapeTensordot_9/MatMulTensordot_9/concat_2*
T0*
Tshape0*
_output_shapes
: 
k
Abs_50/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
U
Abs_50AbsAbs_50/ReadVariableOp*
T0*&
_output_shapes
:
N
	add_139/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_139AddV2Abs_50	add_139/y*
T0*&
_output_shapes
:
H
Log_100Logadd_139*
T0*&
_output_shapes
:
M
Const_81Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_101LogConst_81*
T0*
_output_shapes
: 
X

truediv_50RealDivLog_100Log_101*
T0*&
_output_shapes
:
O
ReadVariableOp_76ReadVariableOpslope*
_output_shapes
: *
dtype0
]
mul_69MulReadVariableOp_76
truediv_50*
T0*&
_output_shapes
:
S
ReadVariableOp_77ReadVariableOp	intercept*
_output_shapes
: *
dtype0
\
add_140AddV2ReadVariableOp_77mul_69*
T0*&
_output_shapes
:
{
differentiable_round_38Roundadd_140*
T0*
_gradient_op_type
Identity*&
_output_shapes
:
t
GreaterEqual_14/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
V
GreaterEqual_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Г
GreaterEqual_14GreaterEqualGreaterEqual_14/ReadVariableOpGreaterEqual_14/y*
T0*&
_output_shapes
:
q
ones_like_28/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
k
ones_like_28/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
W
ones_like_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?

ones_like_28Fillones_like_28/Shapeones_like_28/Const*
T0*&
_output_shapes
:*

index_type0
r
zeros_like_28Const*&
_output_shapes
:*
dtype0*%
valueB*    
v
SelectV2_28SelectV2GreaterEqual_14ones_like_28zeros_like_28*
T0*&
_output_shapes
:
q
LessEqual_14/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
S
LessEqual_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
w
LessEqual_14	LessEqualLessEqual_14/ReadVariableOpLessEqual_14/y*
T0*&
_output_shapes
:
q
ones_like_29/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
k
ones_like_29/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
W
ones_like_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?

ones_like_29Fillones_like_29/Shapeones_like_29/Const*
T0*&
_output_shapes
:*

index_type0
M
mul_70/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
V
mul_70Mulmul_70/xones_like_29*
T0*&
_output_shapes
:
r
zeros_like_29Const*&
_output_shapes
:*
dtype0*%
valueB*    
m
SelectV2_29SelectV2LessEqual_14mul_70zeros_like_29*
T0*&
_output_shapes
:
Y
Add_141AddSelectV2_29SelectV2_28*
T0*&
_output_shapes
:
M
pow_20/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
a
pow_20Powpow_20/xdifferentiable_round_38*
T0*&
_output_shapes
:
O
mul_71MulAdd_141pow_20*
T0*&
_output_shapes
:
o
Reshape_63/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
c
Reshape_63/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
v

Reshape_63ReshapeReshape_63/ReadVariableOpReshape_63/shape*
T0*
Tshape0*
_output_shapes	
:Ц
R
l2_normalize_20/SquareSquare
Reshape_63*
T0*
_output_shapes	
:Ц
_
l2_normalize_20/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_20/SumSuml2_normalize_20/Squarel2_normalize_20/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_20/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_20/MaximumMaximuml2_normalize_20/Suml2_normalize_20/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_20/RsqrtRsqrtl2_normalize_20/Maximum*
T0*
_output_shapes
:
_
l2_normalize_20Mul
Reshape_63l2_normalize_20/Rsqrt*
T0*
_output_shapes	
:Ц
c
Reshape_64/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
c

Reshape_64Reshapemul_71Reshape_64/shape*
T0*
Tshape0*
_output_shapes	
:Ц
R
l2_normalize_21/SquareSquare
Reshape_64*
T0*
_output_shapes	
:Ц
_
l2_normalize_21/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_21/SumSuml2_normalize_21/Squarel2_normalize_21/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_21/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_21/MaximumMaximuml2_normalize_21/Suml2_normalize_21/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_21/RsqrtRsqrtl2_normalize_21/Maximum*
T0*
_output_shapes
:
_
l2_normalize_21Mul
Reshape_64l2_normalize_21/Rsqrt*
T0*
_output_shapes	
:Ц
e
Tensordot_10/transpose/permConst*
_output_shapes
:*
dtype0*
valueB: 
Д
Tensordot_10/transpose	Transposel2_normalize_20Tensordot_10/transpose/perm*
T0*
Tperm0*
_output_shapes	
:Ц
k
Tensordot_10/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   Ц   
Л
Tensordot_10/ReshapeReshapeTensordot_10/transposeTensordot_10/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	Ц
g
Tensordot_10/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB: 
И
Tensordot_10/transpose_1	Transposel2_normalize_21Tensordot_10/transpose_1/perm*
T0*
Tperm0*
_output_shapes	
:Ц
m
Tensordot_10/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"Ц      
С
Tensordot_10/Reshape_1ReshapeTensordot_10/transpose_1Tensordot_10/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	Ц
Ъ
Tensordot_10/MatMulMatMulTensordot_10/ReshapeTensordot_10/Reshape_1*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
U
Tensordot_10/shapeConst*
_output_shapes
: *
dtype0*
valueB 
o
Tensordot_10ReshapeTensordot_10/MatMulTensordot_10/shape*
T0*
Tshape0*
_output_shapes
: 
k
Abs_51/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
U
Abs_51AbsAbs_51/ReadVariableOp*
T0*&
_output_shapes
:
N
	add_142/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_142AddV2Abs_51	add_142/y*
T0*&
_output_shapes
:
H
Log_102Logadd_142*
T0*&
_output_shapes
:
M
Const_82Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_103LogConst_82*
T0*
_output_shapes
: 
X

truediv_51RealDivLog_102Log_103*
T0*&
_output_shapes
:
O
ReadVariableOp_78ReadVariableOpslope*
_output_shapes
: *
dtype0
]
mul_72MulReadVariableOp_78
truediv_51*
T0*&
_output_shapes
:
S
ReadVariableOp_79ReadVariableOp	intercept*
_output_shapes
: *
dtype0
\
add_143AddV2ReadVariableOp_79mul_72*
T0*&
_output_shapes
:
{
differentiable_round_39Roundadd_143*
T0*
_gradient_op_type
Identity*&
_output_shapes
:
t
GreaterEqual_15/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
V
GreaterEqual_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Г
GreaterEqual_15GreaterEqualGreaterEqual_15/ReadVariableOpGreaterEqual_15/y*
T0*&
_output_shapes
:
q
ones_like_30/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
k
ones_like_30/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
W
ones_like_30/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?

ones_like_30Fillones_like_30/Shapeones_like_30/Const*
T0*&
_output_shapes
:*

index_type0
r
zeros_like_30Const*&
_output_shapes
:*
dtype0*%
valueB*    
v
SelectV2_30SelectV2GreaterEqual_15ones_like_30zeros_like_30*
T0*&
_output_shapes
:
q
LessEqual_15/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
S
LessEqual_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
w
LessEqual_15	LessEqualLessEqual_15/ReadVariableOpLessEqual_15/y*
T0*&
_output_shapes
:
q
ones_like_31/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
k
ones_like_31/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
W
ones_like_31/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?

ones_like_31Fillones_like_31/Shapeones_like_31/Const*
T0*&
_output_shapes
:*

index_type0
M
mul_73/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
V
mul_73Mulmul_73/xones_like_31*
T0*&
_output_shapes
:
r
zeros_like_31Const*&
_output_shapes
:*
dtype0*%
valueB*    
m
SelectV2_31SelectV2LessEqual_15mul_73zeros_like_31*
T0*&
_output_shapes
:
Y
Add_144AddSelectV2_31SelectV2_30*
T0*&
_output_shapes
:
M
pow_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
a
pow_21Powpow_21/xdifferentiable_round_39*
T0*&
_output_shapes
:
O
mul_74MulAdd_144pow_21*
T0*&
_output_shapes
:
o
Reshape_65/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
c
Reshape_65/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
v

Reshape_65ReshapeReshape_65/ReadVariableOpReshape_65/shape*
T0*
Tshape0*
_output_shapes	
:Ц
R
l2_normalize_22/SquareSquare
Reshape_65*
T0*
_output_shapes	
:Ц
_
l2_normalize_22/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_22/SumSuml2_normalize_22/Squarel2_normalize_22/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_22/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_22/MaximumMaximuml2_normalize_22/Suml2_normalize_22/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_22/RsqrtRsqrtl2_normalize_22/Maximum*
T0*
_output_shapes
:
_
l2_normalize_22Mul
Reshape_65l2_normalize_22/Rsqrt*
T0*
_output_shapes	
:Ц
c
Reshape_66/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
c

Reshape_66Reshapemul_74Reshape_66/shape*
T0*
Tshape0*
_output_shapes	
:Ц
R
l2_normalize_23/SquareSquare
Reshape_66*
T0*
_output_shapes	
:Ц
_
l2_normalize_23/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_23/SumSuml2_normalize_23/Squarel2_normalize_23/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_23/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_23/MaximumMaximuml2_normalize_23/Suml2_normalize_23/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_23/RsqrtRsqrtl2_normalize_23/Maximum*
T0*
_output_shapes
:
_
l2_normalize_23Mul
Reshape_66l2_normalize_23/Rsqrt*
T0*
_output_shapes	
:Ц
e
Tensordot_11/transpose/permConst*
_output_shapes
:*
dtype0*
valueB: 
Д
Tensordot_11/transpose	Transposel2_normalize_22Tensordot_11/transpose/perm*
T0*
Tperm0*
_output_shapes	
:Ц
k
Tensordot_11/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   Ц   
Л
Tensordot_11/ReshapeReshapeTensordot_11/transposeTensordot_11/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	Ц
g
Tensordot_11/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB: 
И
Tensordot_11/transpose_1	Transposel2_normalize_23Tensordot_11/transpose_1/perm*
T0*
Tperm0*
_output_shapes	
:Ц
m
Tensordot_11/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"Ц      
С
Tensordot_11/Reshape_1ReshapeTensordot_11/transpose_1Tensordot_11/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	Ц
Ъ
Tensordot_11/MatMulMatMulTensordot_11/ReshapeTensordot_11/Reshape_1*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
U
Tensordot_11/shapeConst*
_output_shapes
: *
dtype0*
valueB 
o
Tensordot_11ReshapeTensordot_11/MatMulTensordot_11/shape*
T0*
Tshape0*
_output_shapes
: 
m
Abs_52/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
U
Abs_52AbsAbs_52/ReadVariableOp*
T0*&
_output_shapes
:

N
	add_145/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_145AddV2Abs_52	add_145/y*
T0*&
_output_shapes
:

H
Log_104Logadd_145*
T0*&
_output_shapes
:

M
Const_83Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_105LogConst_83*
T0*
_output_shapes
: 
X

truediv_52RealDivLog_104Log_105*
T0*&
_output_shapes
:

Q
ReadVariableOp_80ReadVariableOpslope_1*
_output_shapes
: *
dtype0
]
mul_75MulReadVariableOp_80
truediv_52*
T0*&
_output_shapes
:

U
ReadVariableOp_81ReadVariableOpintercept_1*
_output_shapes
: *
dtype0
\
add_146AddV2ReadVariableOp_81mul_75*
T0*&
_output_shapes
:

{
differentiable_round_40Roundadd_146*
T0*
_gradient_op_type
Identity*&
_output_shapes
:

v
GreaterEqual_16/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
V
GreaterEqual_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Г
GreaterEqual_16GreaterEqualGreaterEqual_16/ReadVariableOpGreaterEqual_16/y*
T0*&
_output_shapes
:

s
ones_like_32/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
k
ones_like_32/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
W
ones_like_32/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?

ones_like_32Fillones_like_32/Shapeones_like_32/Const*
T0*&
_output_shapes
:
*

index_type0
v
zeros_like_32/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"         
   
X
zeros_like_32/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
М
zeros_like_32Fillzeros_like_32/shape_as_tensorzeros_like_32/Const*
T0*&
_output_shapes
:
*

index_type0
v
SelectV2_32SelectV2GreaterEqual_16ones_like_32zeros_like_32*
T0*&
_output_shapes
:

s
LessEqual_16/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
S
LessEqual_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
w
LessEqual_16	LessEqualLessEqual_16/ReadVariableOpLessEqual_16/y*
T0*&
_output_shapes
:

s
ones_like_33/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
k
ones_like_33/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
W
ones_like_33/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?

ones_like_33Fillones_like_33/Shapeones_like_33/Const*
T0*&
_output_shapes
:
*

index_type0
M
mul_76/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
V
mul_76Mulmul_76/xones_like_33*
T0*&
_output_shapes
:

v
zeros_like_33/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"         
   
X
zeros_like_33/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
М
zeros_like_33Fillzeros_like_33/shape_as_tensorzeros_like_33/Const*
T0*&
_output_shapes
:
*

index_type0
m
SelectV2_33SelectV2LessEqual_16mul_76zeros_like_33*
T0*&
_output_shapes
:

Y
Add_147AddSelectV2_33SelectV2_32*
T0*&
_output_shapes
:

M
pow_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
a
pow_22Powpow_22/xdifferentiable_round_40*
T0*&
_output_shapes
:

O
mul_77MulAdd_147pow_22*
T0*&
_output_shapes
:

q
Reshape_67/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
c
Reshape_67/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
v

Reshape_67ReshapeReshape_67/ReadVariableOpReshape_67/shape*
T0*
Tshape0*
_output_shapes	
:▄
R
l2_normalize_24/SquareSquare
Reshape_67*
T0*
_output_shapes	
:▄
_
l2_normalize_24/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_24/SumSuml2_normalize_24/Squarel2_normalize_24/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_24/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_24/MaximumMaximuml2_normalize_24/Suml2_normalize_24/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_24/RsqrtRsqrtl2_normalize_24/Maximum*
T0*
_output_shapes
:
_
l2_normalize_24Mul
Reshape_67l2_normalize_24/Rsqrt*
T0*
_output_shapes	
:▄
c
Reshape_68/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
c

Reshape_68Reshapemul_77Reshape_68/shape*
T0*
Tshape0*
_output_shapes	
:▄
R
l2_normalize_25/SquareSquare
Reshape_68*
T0*
_output_shapes	
:▄
_
l2_normalize_25/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_25/SumSuml2_normalize_25/Squarel2_normalize_25/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_25/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_25/MaximumMaximuml2_normalize_25/Suml2_normalize_25/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_25/RsqrtRsqrtl2_normalize_25/Maximum*
T0*
_output_shapes
:
_
l2_normalize_25Mul
Reshape_68l2_normalize_25/Rsqrt*
T0*
_output_shapes	
:▄
e
Tensordot_12/transpose/permConst*
_output_shapes
:*
dtype0*
valueB: 
Д
Tensordot_12/transpose	Transposel2_normalize_24Tensordot_12/transpose/perm*
T0*
Tperm0*
_output_shapes	
:▄
k
Tensordot_12/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ▄  
Л
Tensordot_12/ReshapeReshapeTensordot_12/transposeTensordot_12/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	▄
g
Tensordot_12/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB: 
И
Tensordot_12/transpose_1	Transposel2_normalize_25Tensordot_12/transpose_1/perm*
T0*
Tperm0*
_output_shapes	
:▄
m
Tensordot_12/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"▄     
С
Tensordot_12/Reshape_1ReshapeTensordot_12/transpose_1Tensordot_12/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	▄
Ъ
Tensordot_12/MatMulMatMulTensordot_12/ReshapeTensordot_12/Reshape_1*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
U
Tensordot_12/shapeConst*
_output_shapes
: *
dtype0*
valueB 
o
Tensordot_12ReshapeTensordot_12/MatMulTensordot_12/shape*
T0*
Tshape0*
_output_shapes
: 
m
Abs_53/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
U
Abs_53AbsAbs_53/ReadVariableOp*
T0*&
_output_shapes
:

N
	add_148/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_148AddV2Abs_53	add_148/y*
T0*&
_output_shapes
:

H
Log_106Logadd_148*
T0*&
_output_shapes
:

M
Const_84Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_107LogConst_84*
T0*
_output_shapes
: 
X

truediv_53RealDivLog_106Log_107*
T0*&
_output_shapes
:

Q
ReadVariableOp_82ReadVariableOpslope_1*
_output_shapes
: *
dtype0
]
mul_78MulReadVariableOp_82
truediv_53*
T0*&
_output_shapes
:

U
ReadVariableOp_83ReadVariableOpintercept_1*
_output_shapes
: *
dtype0
\
add_149AddV2ReadVariableOp_83mul_78*
T0*&
_output_shapes
:

{
differentiable_round_41Roundadd_149*
T0*
_gradient_op_type
Identity*&
_output_shapes
:

v
GreaterEqual_17/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
V
GreaterEqual_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Г
GreaterEqual_17GreaterEqualGreaterEqual_17/ReadVariableOpGreaterEqual_17/y*
T0*&
_output_shapes
:

s
ones_like_34/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
k
ones_like_34/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
W
ones_like_34/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?

ones_like_34Fillones_like_34/Shapeones_like_34/Const*
T0*&
_output_shapes
:
*

index_type0
v
zeros_like_34/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"         
   
X
zeros_like_34/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
М
zeros_like_34Fillzeros_like_34/shape_as_tensorzeros_like_34/Const*
T0*&
_output_shapes
:
*

index_type0
v
SelectV2_34SelectV2GreaterEqual_17ones_like_34zeros_like_34*
T0*&
_output_shapes
:

s
LessEqual_17/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
S
LessEqual_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
w
LessEqual_17	LessEqualLessEqual_17/ReadVariableOpLessEqual_17/y*
T0*&
_output_shapes
:

s
ones_like_35/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
k
ones_like_35/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   
W
ones_like_35/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?

ones_like_35Fillones_like_35/Shapeones_like_35/Const*
T0*&
_output_shapes
:
*

index_type0
M
mul_79/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
V
mul_79Mulmul_79/xones_like_35*
T0*&
_output_shapes
:

v
zeros_like_35/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"         
   
X
zeros_like_35/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
М
zeros_like_35Fillzeros_like_35/shape_as_tensorzeros_like_35/Const*
T0*&
_output_shapes
:
*

index_type0
m
SelectV2_35SelectV2LessEqual_17mul_79zeros_like_35*
T0*&
_output_shapes
:

Y
Add_150AddSelectV2_35SelectV2_34*
T0*&
_output_shapes
:

M
pow_23/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
a
pow_23Powpow_23/xdifferentiable_round_41*
T0*&
_output_shapes
:

O
mul_80MulAdd_150pow_23*
T0*&
_output_shapes
:

q
Reshape_69/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:
*
dtype0
c
Reshape_69/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
v

Reshape_69ReshapeReshape_69/ReadVariableOpReshape_69/shape*
T0*
Tshape0*
_output_shapes	
:▄
R
l2_normalize_26/SquareSquare
Reshape_69*
T0*
_output_shapes	
:▄
_
l2_normalize_26/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_26/SumSuml2_normalize_26/Squarel2_normalize_26/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_26/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_26/MaximumMaximuml2_normalize_26/Suml2_normalize_26/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_26/RsqrtRsqrtl2_normalize_26/Maximum*
T0*
_output_shapes
:
_
l2_normalize_26Mul
Reshape_69l2_normalize_26/Rsqrt*
T0*
_output_shapes	
:▄
c
Reshape_70/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
c

Reshape_70Reshapemul_80Reshape_70/shape*
T0*
Tshape0*
_output_shapes	
:▄
R
l2_normalize_27/SquareSquare
Reshape_70*
T0*
_output_shapes	
:▄
_
l2_normalize_27/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_27/SumSuml2_normalize_27/Squarel2_normalize_27/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_27/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_27/MaximumMaximuml2_normalize_27/Suml2_normalize_27/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_27/RsqrtRsqrtl2_normalize_27/Maximum*
T0*
_output_shapes
:
_
l2_normalize_27Mul
Reshape_70l2_normalize_27/Rsqrt*
T0*
_output_shapes	
:▄
e
Tensordot_13/transpose/permConst*
_output_shapes
:*
dtype0*
valueB: 
Д
Tensordot_13/transpose	Transposel2_normalize_26Tensordot_13/transpose/perm*
T0*
Tperm0*
_output_shapes	
:▄
k
Tensordot_13/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ▄  
Л
Tensordot_13/ReshapeReshapeTensordot_13/transposeTensordot_13/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	▄
g
Tensordot_13/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB: 
И
Tensordot_13/transpose_1	Transposel2_normalize_27Tensordot_13/transpose_1/perm*
T0*
Tperm0*
_output_shapes	
:▄
m
Tensordot_13/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"▄     
С
Tensordot_13/Reshape_1ReshapeTensordot_13/transpose_1Tensordot_13/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	▄
Ъ
Tensordot_13/MatMulMatMulTensordot_13/ReshapeTensordot_13/Reshape_1*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
U
Tensordot_13/shapeConst*
_output_shapes
: *
dtype0*
valueB 
o
Tensordot_13ReshapeTensordot_13/MatMulTensordot_13/shape*
T0*
Tshape0*
_output_shapes
: 
n
Abs_54/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
V
Abs_54AbsAbs_54/ReadVariableOp*
T0*'
_output_shapes
:1
А
N
	add_151/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
U
add_151AddV2Abs_54	add_151/y*
T0*'
_output_shapes
:1
А
I
Log_108Logadd_151*
T0*'
_output_shapes
:1
А
M
Const_85Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_109LogConst_85*
T0*
_output_shapes
: 
Y

truediv_54RealDivLog_108Log_109*
T0*'
_output_shapes
:1
А
Q
ReadVariableOp_84ReadVariableOpslope_2*
_output_shapes
: *
dtype0
^
mul_81MulReadVariableOp_84
truediv_54*
T0*'
_output_shapes
:1
А
U
ReadVariableOp_85ReadVariableOpintercept_2*
_output_shapes
: *
dtype0
]
add_152AddV2ReadVariableOp_85mul_81*
T0*'
_output_shapes
:1
А
|
differentiable_round_42Roundadd_152*
T0*
_gradient_op_type
Identity*'
_output_shapes
:1
А
w
GreaterEqual_18/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
V
GreaterEqual_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Д
GreaterEqual_18GreaterEqualGreaterEqual_18/ReadVariableOpGreaterEqual_18/y*
T0*'
_output_shapes
:1
А
t
ones_like_36/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
k
ones_like_36/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
W
ones_like_36/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_36Fillones_like_36/Shapeones_like_36/Const*
T0*'
_output_shapes
:1
А*

index_type0
v
zeros_like_36/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
X
zeros_like_36/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_36Fillzeros_like_36/shape_as_tensorzeros_like_36/Const*
T0*'
_output_shapes
:1
А*

index_type0
w
SelectV2_36SelectV2GreaterEqual_18ones_like_36zeros_like_36*
T0*'
_output_shapes
:1
А
t
LessEqual_18/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
S
LessEqual_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
x
LessEqual_18	LessEqualLessEqual_18/ReadVariableOpLessEqual_18/y*
T0*'
_output_shapes
:1
А
t
ones_like_37/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
k
ones_like_37/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
W
ones_like_37/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_37Fillones_like_37/Shapeones_like_37/Const*
T0*'
_output_shapes
:1
А*

index_type0
M
mul_82/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
W
mul_82Mulmul_82/xones_like_37*
T0*'
_output_shapes
:1
А
v
zeros_like_37/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
X
zeros_like_37/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_37Fillzeros_like_37/shape_as_tensorzeros_like_37/Const*
T0*'
_output_shapes
:1
А*

index_type0
n
SelectV2_37SelectV2LessEqual_18mul_82zeros_like_37*
T0*'
_output_shapes
:1
А
Z
Add_153AddSelectV2_37SelectV2_36*
T0*'
_output_shapes
:1
А
M
pow_24/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
b
pow_24Powpow_24/xdifferentiable_round_42*
T0*'
_output_shapes
:1
А
P
mul_83MulAdd_153pow_24*
T0*'
_output_shapes
:1
А
r
Reshape_71/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
c
Reshape_71/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
w

Reshape_71ReshapeReshape_71/ReadVariableOpReshape_71/shape*
T0*
Tshape0*
_output_shapes

:Аъ
S
l2_normalize_28/SquareSquare
Reshape_71*
T0*
_output_shapes

:Аъ
_
l2_normalize_28/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_28/SumSuml2_normalize_28/Squarel2_normalize_28/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_28/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_28/MaximumMaximuml2_normalize_28/Suml2_normalize_28/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_28/RsqrtRsqrtl2_normalize_28/Maximum*
T0*
_output_shapes
:
`
l2_normalize_28Mul
Reshape_71l2_normalize_28/Rsqrt*
T0*
_output_shapes

:Аъ
c
Reshape_72/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
d

Reshape_72Reshapemul_83Reshape_72/shape*
T0*
Tshape0*
_output_shapes

:Аъ
S
l2_normalize_29/SquareSquare
Reshape_72*
T0*
_output_shapes

:Аъ
_
l2_normalize_29/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_29/SumSuml2_normalize_29/Squarel2_normalize_29/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_29/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_29/MaximumMaximuml2_normalize_29/Suml2_normalize_29/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_29/RsqrtRsqrtl2_normalize_29/Maximum*
T0*
_output_shapes
:
`
l2_normalize_29Mul
Reshape_72l2_normalize_29/Rsqrt*
T0*
_output_shapes

:Аъ
e
Tensordot_14/transpose/permConst*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_14/transpose	Transposel2_normalize_28Tensordot_14/transpose/perm*
T0*
Tperm0*
_output_shapes

:Аъ
k
Tensordot_14/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ї  
М
Tensordot_14/ReshapeReshapeTensordot_14/transposeTensordot_14/Reshape/shape*
T0*
Tshape0* 
_output_shapes
:
Аъ
g
Tensordot_14/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB: 
Й
Tensordot_14/transpose_1	Transposel2_normalize_29Tensordot_14/transpose_1/perm*
T0*
Tperm0*
_output_shapes

:Аъ
m
Tensordot_14/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB" ї     
Т
Tensordot_14/Reshape_1ReshapeTensordot_14/transpose_1Tensordot_14/Reshape_1/shape*
T0*
Tshape0* 
_output_shapes
:
Аъ
Ъ
Tensordot_14/MatMulMatMulTensordot_14/ReshapeTensordot_14/Reshape_1*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
U
Tensordot_14/shapeConst*
_output_shapes
: *
dtype0*
valueB 
o
Tensordot_14ReshapeTensordot_14/MatMulTensordot_14/shape*
T0*
Tshape0*
_output_shapes
: 
n
Abs_55/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
V
Abs_55AbsAbs_55/ReadVariableOp*
T0*'
_output_shapes
:1
А
N
	add_154/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
U
add_154AddV2Abs_55	add_154/y*
T0*'
_output_shapes
:1
А
I
Log_110Logadd_154*
T0*'
_output_shapes
:1
А
M
Const_86Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_111LogConst_86*
T0*
_output_shapes
: 
Y

truediv_55RealDivLog_110Log_111*
T0*'
_output_shapes
:1
А
Q
ReadVariableOp_86ReadVariableOpslope_2*
_output_shapes
: *
dtype0
^
mul_84MulReadVariableOp_86
truediv_55*
T0*'
_output_shapes
:1
А
U
ReadVariableOp_87ReadVariableOpintercept_2*
_output_shapes
: *
dtype0
]
add_155AddV2ReadVariableOp_87mul_84*
T0*'
_output_shapes
:1
А
|
differentiable_round_43Roundadd_155*
T0*
_gradient_op_type
Identity*'
_output_shapes
:1
А
w
GreaterEqual_19/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
V
GreaterEqual_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Д
GreaterEqual_19GreaterEqualGreaterEqual_19/ReadVariableOpGreaterEqual_19/y*
T0*'
_output_shapes
:1
А
t
ones_like_38/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
k
ones_like_38/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
W
ones_like_38/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_38Fillones_like_38/Shapeones_like_38/Const*
T0*'
_output_shapes
:1
А*

index_type0
v
zeros_like_38/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
X
zeros_like_38/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_38Fillzeros_like_38/shape_as_tensorzeros_like_38/Const*
T0*'
_output_shapes
:1
А*

index_type0
w
SelectV2_38SelectV2GreaterEqual_19ones_like_38zeros_like_38*
T0*'
_output_shapes
:1
А
t
LessEqual_19/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
S
LessEqual_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
x
LessEqual_19	LessEqualLessEqual_19/ReadVariableOpLessEqual_19/y*
T0*'
_output_shapes
:1
А
t
ones_like_39/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
k
ones_like_39/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
W
ones_like_39/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_39Fillones_like_39/Shapeones_like_39/Const*
T0*'
_output_shapes
:1
А*

index_type0
M
mul_85/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
W
mul_85Mulmul_85/xones_like_39*
T0*'
_output_shapes
:1
А
v
zeros_like_39/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"1   
      А   
X
zeros_like_39/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_39Fillzeros_like_39/shape_as_tensorzeros_like_39/Const*
T0*'
_output_shapes
:1
А*

index_type0
n
SelectV2_39SelectV2LessEqual_19mul_85zeros_like_39*
T0*'
_output_shapes
:1
А
Z
Add_156AddSelectV2_39SelectV2_38*
T0*'
_output_shapes
:1
А
M
pow_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
b
pow_25Powpow_25/xdifferentiable_round_43*
T0*'
_output_shapes
:1
А
P
mul_86MulAdd_156pow_25*
T0*'
_output_shapes
:1
А
r
Reshape_73/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:1
А*
dtype0
c
Reshape_73/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
w

Reshape_73ReshapeReshape_73/ReadVariableOpReshape_73/shape*
T0*
Tshape0*
_output_shapes

:Аъ
S
l2_normalize_30/SquareSquare
Reshape_73*
T0*
_output_shapes

:Аъ
_
l2_normalize_30/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_30/SumSuml2_normalize_30/Squarel2_normalize_30/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_30/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_30/MaximumMaximuml2_normalize_30/Suml2_normalize_30/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_30/RsqrtRsqrtl2_normalize_30/Maximum*
T0*
_output_shapes
:
`
l2_normalize_30Mul
Reshape_73l2_normalize_30/Rsqrt*
T0*
_output_shapes

:Аъ
c
Reshape_74/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
d

Reshape_74Reshapemul_86Reshape_74/shape*
T0*
Tshape0*
_output_shapes

:Аъ
S
l2_normalize_31/SquareSquare
Reshape_74*
T0*
_output_shapes

:Аъ
_
l2_normalize_31/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_31/SumSuml2_normalize_31/Squarel2_normalize_31/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_31/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_31/MaximumMaximuml2_normalize_31/Suml2_normalize_31/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_31/RsqrtRsqrtl2_normalize_31/Maximum*
T0*
_output_shapes
:
`
l2_normalize_31Mul
Reshape_74l2_normalize_31/Rsqrt*
T0*
_output_shapes

:Аъ
e
Tensordot_15/transpose/permConst*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_15/transpose	Transposel2_normalize_30Tensordot_15/transpose/perm*
T0*
Tperm0*
_output_shapes

:Аъ
k
Tensordot_15/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ї  
М
Tensordot_15/ReshapeReshapeTensordot_15/transposeTensordot_15/Reshape/shape*
T0*
Tshape0* 
_output_shapes
:
Аъ
g
Tensordot_15/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB: 
Й
Tensordot_15/transpose_1	Transposel2_normalize_31Tensordot_15/transpose_1/perm*
T0*
Tperm0*
_output_shapes

:Аъ
m
Tensordot_15/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB" ї     
Т
Tensordot_15/Reshape_1ReshapeTensordot_15/transpose_1Tensordot_15/Reshape_1/shape*
T0*
Tshape0* 
_output_shapes
:
Аъ
Ъ
Tensordot_15/MatMulMatMulTensordot_15/ReshapeTensordot_15/Reshape_1*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
U
Tensordot_15/shapeConst*
_output_shapes
: *
dtype0*
valueB 
o
Tensordot_15ReshapeTensordot_15/MatMulTensordot_15/shape*
T0*
Tshape0*
_output_shapes
: 
o
Abs_56/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
W
Abs_56AbsAbs_56/ReadVariableOp*
T0*(
_output_shapes
:АА
N
	add_157/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
V
add_157AddV2Abs_56	add_157/y*
T0*(
_output_shapes
:АА
J
Log_112Logadd_157*
T0*(
_output_shapes
:АА
M
Const_87Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_113LogConst_87*
T0*
_output_shapes
: 
Z

truediv_56RealDivLog_112Log_113*
T0*(
_output_shapes
:АА
Q
ReadVariableOp_88ReadVariableOpslope_4*
_output_shapes
: *
dtype0
_
mul_87MulReadVariableOp_88
truediv_56*
T0*(
_output_shapes
:АА
U
ReadVariableOp_89ReadVariableOpintercept_4*
_output_shapes
: *
dtype0
^
add_158AddV2ReadVariableOp_89mul_87*
T0*(
_output_shapes
:АА
}
differentiable_round_44Roundadd_158*
T0*
_gradient_op_type
Identity*(
_output_shapes
:АА
x
GreaterEqual_20/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
V
GreaterEqual_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Е
GreaterEqual_20GreaterEqualGreaterEqual_20/ReadVariableOpGreaterEqual_20/y*
T0*(
_output_shapes
:АА
u
ones_like_40/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
k
ones_like_40/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
W
ones_like_40/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Б
ones_like_40Fillones_like_40/Shapeones_like_40/Const*
T0*(
_output_shapes
:АА*

index_type0
v
zeros_like_40/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
X
zeros_like_40/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
О
zeros_like_40Fillzeros_like_40/shape_as_tensorzeros_like_40/Const*
T0*(
_output_shapes
:АА*

index_type0
x
SelectV2_40SelectV2GreaterEqual_20ones_like_40zeros_like_40*
T0*(
_output_shapes
:АА
u
LessEqual_20/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
S
LessEqual_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
y
LessEqual_20	LessEqualLessEqual_20/ReadVariableOpLessEqual_20/y*
T0*(
_output_shapes
:АА
u
ones_like_41/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
k
ones_like_41/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
W
ones_like_41/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Б
ones_like_41Fillones_like_41/Shapeones_like_41/Const*
T0*(
_output_shapes
:АА*

index_type0
M
mul_88/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
X
mul_88Mulmul_88/xones_like_41*
T0*(
_output_shapes
:АА
v
zeros_like_41/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
X
zeros_like_41/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
О
zeros_like_41Fillzeros_like_41/shape_as_tensorzeros_like_41/Const*
T0*(
_output_shapes
:АА*

index_type0
o
SelectV2_41SelectV2LessEqual_20mul_88zeros_like_41*
T0*(
_output_shapes
:АА
[
Add_159AddSelectV2_41SelectV2_40*
T0*(
_output_shapes
:АА
M
pow_26/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
c
pow_26Powpow_26/xdifferentiable_round_44*
T0*(
_output_shapes
:АА
Q
mul_89MulAdd_159pow_26*
T0*(
_output_shapes
:АА
s
Reshape_75/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
c
Reshape_75/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
w

Reshape_75ReshapeReshape_75/ReadVariableOpReshape_75/shape*
T0*
Tshape0*
_output_shapes

:АА
S
l2_normalize_32/SquareSquare
Reshape_75*
T0*
_output_shapes

:АА
_
l2_normalize_32/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_32/SumSuml2_normalize_32/Squarel2_normalize_32/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_32/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_32/MaximumMaximuml2_normalize_32/Suml2_normalize_32/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_32/RsqrtRsqrtl2_normalize_32/Maximum*
T0*
_output_shapes
:
`
l2_normalize_32Mul
Reshape_75l2_normalize_32/Rsqrt*
T0*
_output_shapes

:АА
c
Reshape_76/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
d

Reshape_76Reshapemul_89Reshape_76/shape*
T0*
Tshape0*
_output_shapes

:АА
S
l2_normalize_33/SquareSquare
Reshape_76*
T0*
_output_shapes

:АА
_
l2_normalize_33/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_33/SumSuml2_normalize_33/Squarel2_normalize_33/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_33/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_33/MaximumMaximuml2_normalize_33/Suml2_normalize_33/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_33/RsqrtRsqrtl2_normalize_33/Maximum*
T0*
_output_shapes
:
`
l2_normalize_33Mul
Reshape_76l2_normalize_33/Rsqrt*
T0*
_output_shapes

:АА
e
Tensordot_16/transpose/permConst*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_16/transpose	Transposel2_normalize_32Tensordot_16/transpose/perm*
T0*
Tperm0*
_output_shapes

:АА
k
Tensordot_16/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    @  
М
Tensordot_16/ReshapeReshapeTensordot_16/transposeTensordot_16/Reshape/shape*
T0*
Tshape0* 
_output_shapes
:
АА
g
Tensordot_16/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB: 
Й
Tensordot_16/transpose_1	Transposel2_normalize_33Tensordot_16/transpose_1/perm*
T0*
Tperm0*
_output_shapes

:АА
m
Tensordot_16/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB" @     
Т
Tensordot_16/Reshape_1ReshapeTensordot_16/transpose_1Tensordot_16/Reshape_1/shape*
T0*
Tshape0* 
_output_shapes
:
АА
Ъ
Tensordot_16/MatMulMatMulTensordot_16/ReshapeTensordot_16/Reshape_1*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
U
Tensordot_16/shapeConst*
_output_shapes
: *
dtype0*
valueB 
o
Tensordot_16ReshapeTensordot_16/MatMulTensordot_16/shape*
T0*
Tshape0*
_output_shapes
: 
o
Abs_57/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
W
Abs_57AbsAbs_57/ReadVariableOp*
T0*(
_output_shapes
:АА
N
	add_160/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
V
add_160AddV2Abs_57	add_160/y*
T0*(
_output_shapes
:АА
J
Log_114Logadd_160*
T0*(
_output_shapes
:АА
M
Const_88Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_115LogConst_88*
T0*
_output_shapes
: 
Z

truediv_57RealDivLog_114Log_115*
T0*(
_output_shapes
:АА
Q
ReadVariableOp_90ReadVariableOpslope_4*
_output_shapes
: *
dtype0
_
mul_90MulReadVariableOp_90
truediv_57*
T0*(
_output_shapes
:АА
U
ReadVariableOp_91ReadVariableOpintercept_4*
_output_shapes
: *
dtype0
^
add_161AddV2ReadVariableOp_91mul_90*
T0*(
_output_shapes
:АА
}
differentiable_round_45Roundadd_161*
T0*
_gradient_op_type
Identity*(
_output_shapes
:АА
x
GreaterEqual_21/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
V
GreaterEqual_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Е
GreaterEqual_21GreaterEqualGreaterEqual_21/ReadVariableOpGreaterEqual_21/y*
T0*(
_output_shapes
:АА
u
ones_like_42/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
k
ones_like_42/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
W
ones_like_42/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Б
ones_like_42Fillones_like_42/Shapeones_like_42/Const*
T0*(
_output_shapes
:АА*

index_type0
v
zeros_like_42/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
X
zeros_like_42/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
О
zeros_like_42Fillzeros_like_42/shape_as_tensorzeros_like_42/Const*
T0*(
_output_shapes
:АА*

index_type0
x
SelectV2_42SelectV2GreaterEqual_21ones_like_42zeros_like_42*
T0*(
_output_shapes
:АА
u
LessEqual_21/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
S
LessEqual_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
y
LessEqual_21	LessEqualLessEqual_21/ReadVariableOpLessEqual_21/y*
T0*(
_output_shapes
:АА
u
ones_like_43/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
k
ones_like_43/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
W
ones_like_43/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Б
ones_like_43Fillones_like_43/Shapeones_like_43/Const*
T0*(
_output_shapes
:АА*

index_type0
M
mul_91/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
X
mul_91Mulmul_91/xones_like_43*
T0*(
_output_shapes
:АА
v
zeros_like_43/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         А   
X
zeros_like_43/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
О
zeros_like_43Fillzeros_like_43/shape_as_tensorzeros_like_43/Const*
T0*(
_output_shapes
:АА*

index_type0
o
SelectV2_43SelectV2LessEqual_21mul_91zeros_like_43*
T0*(
_output_shapes
:АА
[
Add_162AddSelectV2_43SelectV2_42*
T0*(
_output_shapes
:АА
M
pow_27/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
c
pow_27Powpow_27/xdifferentiable_round_45*
T0*(
_output_shapes
:АА
Q
mul_92MulAdd_162pow_27*
T0*(
_output_shapes
:АА
s
Reshape_77/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
c
Reshape_77/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
w

Reshape_77ReshapeReshape_77/ReadVariableOpReshape_77/shape*
T0*
Tshape0*
_output_shapes

:АА
S
l2_normalize_34/SquareSquare
Reshape_77*
T0*
_output_shapes

:АА
_
l2_normalize_34/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_34/SumSuml2_normalize_34/Squarel2_normalize_34/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_34/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_34/MaximumMaximuml2_normalize_34/Suml2_normalize_34/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_34/RsqrtRsqrtl2_normalize_34/Maximum*
T0*
_output_shapes
:
`
l2_normalize_34Mul
Reshape_77l2_normalize_34/Rsqrt*
T0*
_output_shapes

:АА
c
Reshape_78/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
d

Reshape_78Reshapemul_92Reshape_78/shape*
T0*
Tshape0*
_output_shapes

:АА
S
l2_normalize_35/SquareSquare
Reshape_78*
T0*
_output_shapes

:АА
_
l2_normalize_35/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_35/SumSuml2_normalize_35/Squarel2_normalize_35/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_35/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_35/MaximumMaximuml2_normalize_35/Suml2_normalize_35/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_35/RsqrtRsqrtl2_normalize_35/Maximum*
T0*
_output_shapes
:
`
l2_normalize_35Mul
Reshape_78l2_normalize_35/Rsqrt*
T0*
_output_shapes

:АА
e
Tensordot_17/transpose/permConst*
_output_shapes
:*
dtype0*
valueB: 
Е
Tensordot_17/transpose	Transposel2_normalize_34Tensordot_17/transpose/perm*
T0*
Tperm0*
_output_shapes

:АА
k
Tensordot_17/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    @  
М
Tensordot_17/ReshapeReshapeTensordot_17/transposeTensordot_17/Reshape/shape*
T0*
Tshape0* 
_output_shapes
:
АА
g
Tensordot_17/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB: 
Й
Tensordot_17/transpose_1	Transposel2_normalize_35Tensordot_17/transpose_1/perm*
T0*
Tperm0*
_output_shapes

:АА
m
Tensordot_17/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB" @     
Т
Tensordot_17/Reshape_1ReshapeTensordot_17/transpose_1Tensordot_17/Reshape_1/shape*
T0*
Tshape0* 
_output_shapes
:
АА
Ъ
Tensordot_17/MatMulMatMulTensordot_17/ReshapeTensordot_17/Reshape_1*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
U
Tensordot_17/shapeConst*
_output_shapes
: *
dtype0*
valueB 
o
Tensordot_17ReshapeTensordot_17/MatMulTensordot_17/shape*
T0*
Tshape0*
_output_shapes
: 
n
Abs_58/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
V
Abs_58AbsAbs_58/ReadVariableOp*
T0*'
_output_shapes
:А

N
	add_163/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
U
add_163AddV2Abs_58	add_163/y*
T0*'
_output_shapes
:А

I
Log_116Logadd_163*
T0*'
_output_shapes
:А

M
Const_89Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_117LogConst_89*
T0*
_output_shapes
: 
Y

truediv_58RealDivLog_116Log_117*
T0*'
_output_shapes
:А

Q
ReadVariableOp_92ReadVariableOpslope_5*
_output_shapes
: *
dtype0
^
mul_93MulReadVariableOp_92
truediv_58*
T0*'
_output_shapes
:А

U
ReadVariableOp_93ReadVariableOpintercept_5*
_output_shapes
: *
dtype0
]
add_164AddV2ReadVariableOp_93mul_93*
T0*'
_output_shapes
:А

|
differentiable_round_46Roundadd_164*
T0*
_gradient_op_type
Identity*'
_output_shapes
:А

w
GreaterEqual_22/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
V
GreaterEqual_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Д
GreaterEqual_22GreaterEqualGreaterEqual_22/ReadVariableOpGreaterEqual_22/y*
T0*'
_output_shapes
:А

t
ones_like_44/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
k
ones_like_44/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
W
ones_like_44/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_44Fillones_like_44/Shapeones_like_44/Const*
T0*'
_output_shapes
:А
*

index_type0
v
zeros_like_44/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
X
zeros_like_44/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_44Fillzeros_like_44/shape_as_tensorzeros_like_44/Const*
T0*'
_output_shapes
:А
*

index_type0
w
SelectV2_44SelectV2GreaterEqual_22ones_like_44zeros_like_44*
T0*'
_output_shapes
:А

t
LessEqual_22/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
S
LessEqual_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
x
LessEqual_22	LessEqualLessEqual_22/ReadVariableOpLessEqual_22/y*
T0*'
_output_shapes
:А

t
ones_like_45/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
k
ones_like_45/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
W
ones_like_45/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_45Fillones_like_45/Shapeones_like_45/Const*
T0*'
_output_shapes
:А
*

index_type0
M
mul_94/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
W
mul_94Mulmul_94/xones_like_45*
T0*'
_output_shapes
:А

v
zeros_like_45/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
X
zeros_like_45/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_45Fillzeros_like_45/shape_as_tensorzeros_like_45/Const*
T0*'
_output_shapes
:А
*

index_type0
n
SelectV2_45SelectV2LessEqual_22mul_94zeros_like_45*
T0*'
_output_shapes
:А

Z
Add_165AddSelectV2_45SelectV2_44*
T0*'
_output_shapes
:А

M
pow_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
b
pow_28Powpow_28/xdifferentiable_round_46*
T0*'
_output_shapes
:А

P
mul_95MulAdd_165pow_28*
T0*'
_output_shapes
:А

r
Reshape_79/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
c
Reshape_79/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
v

Reshape_79ReshapeReshape_79/ReadVariableOpReshape_79/shape*
T0*
Tshape0*
_output_shapes	
:А

R
l2_normalize_36/SquareSquare
Reshape_79*
T0*
_output_shapes	
:А

_
l2_normalize_36/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_36/SumSuml2_normalize_36/Squarel2_normalize_36/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_36/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_36/MaximumMaximuml2_normalize_36/Suml2_normalize_36/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_36/RsqrtRsqrtl2_normalize_36/Maximum*
T0*
_output_shapes
:
_
l2_normalize_36Mul
Reshape_79l2_normalize_36/Rsqrt*
T0*
_output_shapes	
:А

c
Reshape_80/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
c

Reshape_80Reshapemul_95Reshape_80/shape*
T0*
Tshape0*
_output_shapes	
:А

R
l2_normalize_37/SquareSquare
Reshape_80*
T0*
_output_shapes	
:А

_
l2_normalize_37/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_37/SumSuml2_normalize_37/Squarel2_normalize_37/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_37/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_37/MaximumMaximuml2_normalize_37/Suml2_normalize_37/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_37/RsqrtRsqrtl2_normalize_37/Maximum*
T0*
_output_shapes
:
_
l2_normalize_37Mul
Reshape_80l2_normalize_37/Rsqrt*
T0*
_output_shapes	
:А

e
Tensordot_18/transpose/permConst*
_output_shapes
:*
dtype0*
valueB: 
Д
Tensordot_18/transpose	Transposel2_normalize_36Tensordot_18/transpose/perm*
T0*
Tperm0*
_output_shapes	
:А

k
Tensordot_18/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
Л
Tensordot_18/ReshapeReshapeTensordot_18/transposeTensordot_18/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	А

g
Tensordot_18/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB: 
И
Tensordot_18/transpose_1	Transposel2_normalize_37Tensordot_18/transpose_1/perm*
T0*
Tperm0*
_output_shapes	
:А

m
Tensordot_18/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
С
Tensordot_18/Reshape_1ReshapeTensordot_18/transpose_1Tensordot_18/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	А

Ъ
Tensordot_18/MatMulMatMulTensordot_18/ReshapeTensordot_18/Reshape_1*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
U
Tensordot_18/shapeConst*
_output_shapes
: *
dtype0*
valueB 
o
Tensordot_18ReshapeTensordot_18/MatMulTensordot_18/shape*
T0*
Tshape0*
_output_shapes
: 
n
Abs_59/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
V
Abs_59AbsAbs_59/ReadVariableOp*
T0*'
_output_shapes
:А

N
	add_166/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
U
add_166AddV2Abs_59	add_166/y*
T0*'
_output_shapes
:А

I
Log_118Logadd_166*
T0*'
_output_shapes
:А

M
Const_90Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_119LogConst_90*
T0*
_output_shapes
: 
Y

truediv_59RealDivLog_118Log_119*
T0*'
_output_shapes
:А

Q
ReadVariableOp_94ReadVariableOpslope_5*
_output_shapes
: *
dtype0
^
mul_96MulReadVariableOp_94
truediv_59*
T0*'
_output_shapes
:А

U
ReadVariableOp_95ReadVariableOpintercept_5*
_output_shapes
: *
dtype0
]
add_167AddV2ReadVariableOp_95mul_96*
T0*'
_output_shapes
:А

|
differentiable_round_47Roundadd_167*
T0*
_gradient_op_type
Identity*'
_output_shapes
:А

w
GreaterEqual_23/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
V
GreaterEqual_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
Д
GreaterEqual_23GreaterEqualGreaterEqual_23/ReadVariableOpGreaterEqual_23/y*
T0*'
_output_shapes
:А

t
ones_like_46/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
k
ones_like_46/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
W
ones_like_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_46Fillones_like_46/Shapeones_like_46/Const*
T0*'
_output_shapes
:А
*

index_type0
v
zeros_like_46/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
X
zeros_like_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_46Fillzeros_like_46/shape_as_tensorzeros_like_46/Const*
T0*'
_output_shapes
:А
*

index_type0
w
SelectV2_46SelectV2GreaterEqual_23ones_like_46zeros_like_46*
T0*'
_output_shapes
:А

t
LessEqual_23/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
S
LessEqual_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж╡
x
LessEqual_23	LessEqualLessEqual_23/ReadVariableOpLessEqual_23/y*
T0*'
_output_shapes
:А

t
ones_like_47/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
k
ones_like_47/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
W
ones_like_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
А
ones_like_47Fillones_like_47/Shapeones_like_47/Const*
T0*'
_output_shapes
:А
*

index_type0
M
mul_97/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐
W
mul_97Mulmul_97/xones_like_47*
T0*'
_output_shapes
:А

v
zeros_like_47/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"А         
   
X
zeros_like_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Н
zeros_like_47Fillzeros_like_47/shape_as_tensorzeros_like_47/Const*
T0*'
_output_shapes
:А
*

index_type0
n
SelectV2_47SelectV2LessEqual_23mul_97zeros_like_47*
T0*'
_output_shapes
:А

Z
Add_168AddSelectV2_47SelectV2_46*
T0*'
_output_shapes
:А

M
pow_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
b
pow_29Powpow_29/xdifferentiable_round_47*
T0*'
_output_shapes
:А

P
mul_98MulAdd_168pow_29*
T0*'
_output_shapes
:А

r
Reshape_81/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А
*
dtype0
c
Reshape_81/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
v

Reshape_81ReshapeReshape_81/ReadVariableOpReshape_81/shape*
T0*
Tshape0*
_output_shapes	
:А

R
l2_normalize_38/SquareSquare
Reshape_81*
T0*
_output_shapes	
:А

_
l2_normalize_38/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_38/SumSuml2_normalize_38/Squarel2_normalize_38/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_38/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_38/MaximumMaximuml2_normalize_38/Suml2_normalize_38/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_38/RsqrtRsqrtl2_normalize_38/Maximum*
T0*
_output_shapes
:
_
l2_normalize_38Mul
Reshape_81l2_normalize_38/Rsqrt*
T0*
_output_shapes	
:А

c
Reshape_82/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
c

Reshape_82Reshapemul_98Reshape_82/shape*
T0*
Tshape0*
_output_shapes	
:А

R
l2_normalize_39/SquareSquare
Reshape_82*
T0*
_output_shapes	
:А

_
l2_normalize_39/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Л
l2_normalize_39/SumSuml2_normalize_39/Squarel2_normalize_39/Const*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
^
l2_normalize_39/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝М+
w
l2_normalize_39/MaximumMaximuml2_normalize_39/Suml2_normalize_39/Maximum/y*
T0*
_output_shapes
:
\
l2_normalize_39/RsqrtRsqrtl2_normalize_39/Maximum*
T0*
_output_shapes
:
_
l2_normalize_39Mul
Reshape_82l2_normalize_39/Rsqrt*
T0*
_output_shapes	
:А

e
Tensordot_19/transpose/permConst*
_output_shapes
:*
dtype0*
valueB: 
Д
Tensordot_19/transpose	Transposel2_normalize_38Tensordot_19/transpose/perm*
T0*
Tperm0*
_output_shapes	
:А

k
Tensordot_19/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
Л
Tensordot_19/ReshapeReshapeTensordot_19/transposeTensordot_19/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	А

g
Tensordot_19/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB: 
И
Tensordot_19/transpose_1	Transposel2_normalize_39Tensordot_19/transpose_1/perm*
T0*
Tperm0*
_output_shapes	
:А

m
Tensordot_19/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
С
Tensordot_19/Reshape_1ReshapeTensordot_19/transpose_1Tensordot_19/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	А

Ъ
Tensordot_19/MatMulMatMulTensordot_19/ReshapeTensordot_19/Reshape_1*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
U
Tensordot_19/shapeConst*
_output_shapes
: *
dtype0*
valueB 
o
Tensordot_19ReshapeTensordot_19/MatMulTensordot_19/shape*
T0*
Tshape0*
_output_shapes
: 
А
Rank_27/packedPackadd_61add_68add_75add_82add_89add_96*
N*
T0*
_output_shapes
:*

axis 
I
Rank_27Const*
_output_shapes
: *
dtype0*
value	B :
P
range_27/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_27/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_27Rangerange_27/startRank_27range_27/delta*

Tidx0*
_output_shapes
:
~
Mean_4/inputPackadd_61add_68add_75add_82add_89add_96*
N*
T0*
_output_shapes
:*

axis 
d
Mean_4MeanMean_4/inputrange_27*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
b
Rank_28/packed/0Packnorm_1/Squeeze*
N*
T0*
_output_shapes
:*

axis 
b
Rank_28/packed/1Packnorm_3/Squeeze*
N*
T0*
_output_shapes
:*

axis 
b
Rank_28/packed/2Packnorm_5/Squeeze*
N*
T0*
_output_shapes
:*

axis 
b
Rank_28/packed/3Packnorm_7/Squeeze*
N*
T0*
_output_shapes
:*

axis 
b
Rank_28/packed/4Packnorm_9/Squeeze*
N*
T0*
_output_shapes
:*

axis 
о
Rank_28/packedPackRank_28/packed/0Rank_28/packed/1Rank_28/packed/2Rank_28/packed/3Rank_28/packed/4*
N*
T0*
_output_shapes

:*

axis 
I
Rank_28Const*
_output_shapes
: *
dtype0*
value	B :
P
range_28/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_28/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_28Rangerange_28/startRank_28range_28/delta*

Tidx0*
_output_shapes
:
`
Mean_5/input/0Packnorm_1/Squeeze*
N*
T0*
_output_shapes
:*

axis 
`
Mean_5/input/1Packnorm_3/Squeeze*
N*
T0*
_output_shapes
:*

axis 
`
Mean_5/input/2Packnorm_5/Squeeze*
N*
T0*
_output_shapes
:*

axis 
`
Mean_5/input/3Packnorm_7/Squeeze*
N*
T0*
_output_shapes
:*

axis 
`
Mean_5/input/4Packnorm_9/Squeeze*
N*
T0*
_output_shapes
:*

axis 
в
Mean_5/inputPackMean_5/input/0Mean_5/input/1Mean_5/input/2Mean_5/input/3Mean_5/input/4*
N*
T0*
_output_shapes

:*

axis 
d
Mean_5MeanMean_5/inputrange_28*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
M
mul_99/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
I
mul_99Mulnorm_11/Squeezemul_99/y*
T0*
_output_shapes
: 
N
	mul_100/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
K
mul_100Mulnorm_13/Squeeze	mul_100/y*
T0*
_output_shapes
: 
N
	mul_101/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
K
mul_101Mulnorm_15/Squeeze	mul_101/y*
T0*
_output_shapes
: 
N
	mul_102/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
K
mul_102Mulnorm_17/Squeeze	mul_102/y*
T0*
_output_shapes
: 
N
	mul_103/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
K
mul_103Mulnorm_19/Squeeze	mul_103/y*
T0*
_output_shapes
: 
е
Rank_29/packedPacknorm_11/Squeezenorm_13/Squeezenorm_15/Squeezenorm_17/Squeezenorm_19/Squeeze*
N*
T0*
_output_shapes
:*

axis 
I
Rank_29Const*
_output_shapes
: *
dtype0*
value	B :
P
range_29/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_29/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_29Rangerange_29/startRank_29range_29/delta*

Tidx0*
_output_shapes
:
г
Mean_6/inputPacknorm_11/Squeezenorm_13/Squeezenorm_15/Squeezenorm_17/Squeezenorm_19/Squeeze*
N*
T0*
_output_shapes
:*

axis 
d
Mean_6MeanMean_6/inputrange_29*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
_
Rank_30/packed/0PackTensordot_1*
N*
T0*
_output_shapes
:*

axis 
_
Rank_30/packed/1PackTensordot_3*
N*
T0*
_output_shapes
:*

axis 
_
Rank_30/packed/2PackTensordot_5*
N*
T0*
_output_shapes
:*

axis 
_
Rank_30/packed/3PackTensordot_7*
N*
T0*
_output_shapes
:*

axis 
_
Rank_30/packed/4PackTensordot_9*
N*
T0*
_output_shapes
:*

axis 
о
Rank_30/packedPackRank_30/packed/0Rank_30/packed/1Rank_30/packed/2Rank_30/packed/3Rank_30/packed/4*
N*
T0*
_output_shapes

:*

axis 
I
Rank_30Const*
_output_shapes
: *
dtype0*
value	B :
P
range_30/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_30/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_30Rangerange_30/startRank_30range_30/delta*

Tidx0*
_output_shapes
:
]
Mean_7/input/0PackTensordot_1*
N*
T0*
_output_shapes
:*

axis 
]
Mean_7/input/1PackTensordot_3*
N*
T0*
_output_shapes
:*

axis 
]
Mean_7/input/2PackTensordot_5*
N*
T0*
_output_shapes
:*

axis 
]
Mean_7/input/3PackTensordot_7*
N*
T0*
_output_shapes
:*

axis 
]
Mean_7/input/4PackTensordot_9*
N*
T0*
_output_shapes
:*

axis 
в
Mean_7/inputPackMean_7/input/0Mean_7/input/1Mean_7/input/2Mean_7/input/3Mean_7/input/4*
N*
T0*
_output_shapes

:*

axis 
d
Mean_7MeanMean_7/inputrange_30*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Ц
Rank_31/packedPackTensordot_11Tensordot_13Tensordot_15Tensordot_17Tensordot_19*
N*
T0*
_output_shapes
:*

axis 
I
Rank_31Const*
_output_shapes
: *
dtype0*
value	B :
P
range_31/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_31/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_31Rangerange_31/startRank_31range_31/delta*

Tidx0*
_output_shapes
:
Ф
Mean_8/inputPackTensordot_11Tensordot_13Tensordot_15Tensordot_17Tensordot_19*
N*
T0*
_output_shapes
:*

axis 
d
Mean_8MeanMean_8/inputrange_31*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
z
hp_cross_entropy_flatten_1/tagsConst*
_output_shapes
: *
dtype0*+
value"B  Bhp_cross_entropy_flatten_1
s
hp_cross_entropy_flatten_1ScalarSummaryhp_cross_entropy_flatten_1/tagsMean*
T0*
_output_shapes
: 
|
 distillation_loss_flatten_1/tagsConst*
_output_shapes
: *
dtype0*,
value#B! Bdistillation_loss_flatten_1
w
distillation_loss_flatten_1ScalarSummary distillation_loss_flatten_1/tagsMean_1*
T0*
_output_shapes
: 
d
b_i_t___l_o_s_s/tagsConst*
_output_shapes
: *
dtype0* 
valueB Bb_i_t___l_o_s_s
^
b_i_t___l_o_s_sScalarSummaryb_i_t___l_o_s_s/tagsSum_2*
T0*
_output_shapes
: 
Р
*r_e_g_u_l_a_r_i_z_a_t_i_o_n___t_e_r_m/tagsConst*
_output_shapes
: *
dtype0*6
value-B+ B%r_e_g_u_l_a_r_i_z_a_t_i_o_n___t_e_r_m
Л
%r_e_g_u_l_a_r_i_z_a_t_i_o_n___t_e_r_mScalarSummary*r_e_g_u_l_a_r_i_z_a_t_i_o_n___t_e_r_m/tagsadd_46*
T0*
_output_shapes
: 
p
lp_accuracy_flatten_1/tagsConst*
_output_shapes
: *
dtype0*&
valueB Blp_accuracy_flatten_1
k
lp_accuracy_flatten_1ScalarSummarylp_accuracy_flatten_1/tagsMean_2*
T0*
_output_shapes
: 
p
hp_accuracy_flatten_1/tagsConst*
_output_shapes
: *
dtype0*&
valueB Bhp_accuracy_flatten_1
k
hp_accuracy_flatten_1ScalarSummaryhp_accuracy_flatten_1/tagsMean_3*
T0*
_output_shapes
: 
p
total_loss_total_loss/tagsConst*
_output_shapes
: *
dtype0*&
valueB Btotal_loss_total_loss
k
total_loss_total_lossScalarSummarytotal_loss_total_loss/tagsadd_54*
T0*
_output_shapes
: 
l
weights_bits_conv2d/tagsConst*
_output_shapes
: *
dtype0*$
valueB Bweights_bits_conv2d
g
weights_bits_conv2dScalarSummaryweights_bits_conv2d/tagsadd_61*
T0*
_output_shapes
: 
p
weights_bits_conv2d_1/tagsConst*
_output_shapes
: *
dtype0*&
valueB Bweights_bits_conv2d_1
k
weights_bits_conv2d_1ScalarSummaryweights_bits_conv2d_1/tagsadd_68*
T0*
_output_shapes
: 
p
weights_bits_conv2d_2/tagsConst*
_output_shapes
: *
dtype0*&
valueB Bweights_bits_conv2d_2
k
weights_bits_conv2d_2ScalarSummaryweights_bits_conv2d_2/tagsadd_75*
T0*
_output_shapes
: 
x
weights_bits_activation_3/tagsConst*
_output_shapes
: *
dtype0**
value!B Bweights_bits_activation_3
s
weights_bits_activation_3ScalarSummaryweights_bits_activation_3/tagsadd_82*
T0*
_output_shapes
: 
p
weights_bits_conv2d_3/tagsConst*
_output_shapes
: *
dtype0*&
valueB Bweights_bits_conv2d_3
k
weights_bits_conv2d_3ScalarSummaryweights_bits_conv2d_3/tagsadd_89*
T0*
_output_shapes
: 
p
weights_bits_conv2d_4/tagsConst*
_output_shapes
: *
dtype0*&
valueB Bweights_bits_conv2d_4
k
weights_bits_conv2d_4ScalarSummaryweights_bits_conv2d_4/tagsadd_96*
T0*
_output_shapes
: 
^
bits_average/tagsConst*
_output_shapes
: *
dtype0*
valueB Bbits_average
Y
bits_averageScalarSummarybits_average/tagsMean_4*
T0*
_output_shapes
: 
j
qerr_op_activation/tagsConst*
_output_shapes
: *
dtype0*#
valueB Bqerr_op_activation
m
qerr_op_activationScalarSummaryqerr_op_activation/tagsnorm_1/Squeeze*
T0*
_output_shapes
: 
n
qerr_op_activation_1/tagsConst*
_output_shapes
: *
dtype0*%
valueB Bqerr_op_activation_1
q
qerr_op_activation_1ScalarSummaryqerr_op_activation_1/tagsnorm_3/Squeeze*
T0*
_output_shapes
: 
n
qerr_op_activation_2/tagsConst*
_output_shapes
: *
dtype0*%
valueB Bqerr_op_activation_2
q
qerr_op_activation_2ScalarSummaryqerr_op_activation_2/tagsnorm_5/Squeeze*
T0*
_output_shapes
: 
n
qerr_op_activation_3/tagsConst*
_output_shapes
: *
dtype0*%
valueB Bqerr_op_activation_3
q
qerr_op_activation_3ScalarSummaryqerr_op_activation_3/tagsnorm_7/Squeeze*
T0*
_output_shapes
: 
n
qerr_op_activation_4/tagsConst*
_output_shapes
: *
dtype0*%
valueB Bqerr_op_activation_4
q
qerr_op_activation_4ScalarSummaryqerr_op_activation_4/tagsnorm_9/Squeeze*
T0*
_output_shapes
: 
d
qerr_op_average/tagsConst*
_output_shapes
: *
dtype0* 
valueB Bqerr_op_average
_
qerr_op_averageScalarSummaryqerr_op_average/tagsMean_5*
T0*
_output_shapes
: 
v
qerr_weights_bias_conv2d/tagsConst*
_output_shapes
: *
dtype0*)
value B Bqerr_weights_bias_conv2d
q
qerr_weights_bias_conv2dScalarSummaryqerr_weights_bias_conv2d/tagsmul_99*
T0*
_output_shapes
: 
z
qerr_weights_bias_conv2d_1/tagsConst*
_output_shapes
: *
dtype0*+
value"B  Bqerr_weights_bias_conv2d_1
v
qerr_weights_bias_conv2d_1ScalarSummaryqerr_weights_bias_conv2d_1/tagsmul_100*
T0*
_output_shapes
: 
z
qerr_weights_bias_conv2d_2/tagsConst*
_output_shapes
: *
dtype0*+
value"B  Bqerr_weights_bias_conv2d_2
v
qerr_weights_bias_conv2d_2ScalarSummaryqerr_weights_bias_conv2d_2/tagsmul_101*
T0*
_output_shapes
: 
z
qerr_weights_bias_conv2d_3/tagsConst*
_output_shapes
: *
dtype0*+
value"B  Bqerr_weights_bias_conv2d_3
v
qerr_weights_bias_conv2d_3ScalarSummaryqerr_weights_bias_conv2d_3/tagsmul_102*
T0*
_output_shapes
: 
z
qerr_weights_bias_conv2d_4/tagsConst*
_output_shapes
: *
dtype0*+
value"B  Bqerr_weights_bias_conv2d_4
v
qerr_weights_bias_conv2d_4ScalarSummaryqerr_weights_bias_conv2d_4/tagsmul_103*
T0*
_output_shapes
: 
|
 qerr_weights_weights_conv2d/tagsConst*
_output_shapes
: *
dtype0*,
value#B! Bqerr_weights_weights_conv2d
А
qerr_weights_weights_conv2dScalarSummary qerr_weights_weights_conv2d/tagsnorm_11/Squeeze*
T0*
_output_shapes
: 
А
"qerr_weights_weights_conv2d_1/tagsConst*
_output_shapes
: *
dtype0*.
value%B# Bqerr_weights_weights_conv2d_1
Д
qerr_weights_weights_conv2d_1ScalarSummary"qerr_weights_weights_conv2d_1/tagsnorm_13/Squeeze*
T0*
_output_shapes
: 
А
"qerr_weights_weights_conv2d_2/tagsConst*
_output_shapes
: *
dtype0*.
value%B# Bqerr_weights_weights_conv2d_2
Д
qerr_weights_weights_conv2d_2ScalarSummary"qerr_weights_weights_conv2d_2/tagsnorm_15/Squeeze*
T0*
_output_shapes
: 
А
"qerr_weights_weights_conv2d_3/tagsConst*
_output_shapes
: *
dtype0*.
value%B# Bqerr_weights_weights_conv2d_3
Д
qerr_weights_weights_conv2d_3ScalarSummary"qerr_weights_weights_conv2d_3/tagsnorm_17/Squeeze*
T0*
_output_shapes
: 
А
"qerr_weights_weights_conv2d_4/tagsConst*
_output_shapes
: *
dtype0*.
value%B# Bqerr_weights_weights_conv2d_4
Д
qerr_weights_weights_conv2d_4ScalarSummary"qerr_weights_weights_conv2d_4/tagsnorm_19/Squeeze*
T0*
_output_shapes
: 
n
qerr_weights_average/tagsConst*
_output_shapes
: *
dtype0*%
valueB Bqerr_weights_average
i
qerr_weights_averageScalarSummaryqerr_weights_average/tagsMean_6*
T0*
_output_shapes
: 
n
qangle_op_activation/tagsConst*
_output_shapes
: *
dtype0*%
valueB Bqangle_op_activation
n
qangle_op_activationScalarSummaryqangle_op_activation/tagsTensordot_1*
T0*
_output_shapes
: 
r
qangle_op_activation_1/tagsConst*
_output_shapes
: *
dtype0*'
valueB Bqangle_op_activation_1
r
qangle_op_activation_1ScalarSummaryqangle_op_activation_1/tagsTensordot_3*
T0*
_output_shapes
: 
r
qangle_op_activation_2/tagsConst*
_output_shapes
: *
dtype0*'
valueB Bqangle_op_activation_2
r
qangle_op_activation_2ScalarSummaryqangle_op_activation_2/tagsTensordot_5*
T0*
_output_shapes
: 
r
qangle_op_activation_3/tagsConst*
_output_shapes
: *
dtype0*'
valueB Bqangle_op_activation_3
r
qangle_op_activation_3ScalarSummaryqangle_op_activation_3/tagsTensordot_7*
T0*
_output_shapes
: 
r
qangle_op_activation_4/tagsConst*
_output_shapes
: *
dtype0*'
valueB Bqangle_op_activation_4
r
qangle_op_activation_4ScalarSummaryqangle_op_activation_4/tagsTensordot_9*
T0*
_output_shapes
: 
h
qangle_op_average/tagsConst*
_output_shapes
: *
dtype0*"
valueB Bqangle_op_average
c
qangle_op_averageScalarSummaryqangle_op_average/tagsMean_7*
T0*
_output_shapes
: 
А
"qangle_weights_weights_conv2d/tagsConst*
_output_shapes
: *
dtype0*.
value%B# Bqangle_weights_weights_conv2d
Б
qangle_weights_weights_conv2dScalarSummary"qangle_weights_weights_conv2d/tagsTensordot_11*
T0*
_output_shapes
: 
Д
$qangle_weights_weights_conv2d_1/tagsConst*
_output_shapes
: *
dtype0*0
value'B% Bqangle_weights_weights_conv2d_1
Е
qangle_weights_weights_conv2d_1ScalarSummary$qangle_weights_weights_conv2d_1/tagsTensordot_13*
T0*
_output_shapes
: 
Д
$qangle_weights_weights_conv2d_2/tagsConst*
_output_shapes
: *
dtype0*0
value'B% Bqangle_weights_weights_conv2d_2
Е
qangle_weights_weights_conv2d_2ScalarSummary$qangle_weights_weights_conv2d_2/tagsTensordot_15*
T0*
_output_shapes
: 
Д
$qangle_weights_weights_conv2d_3/tagsConst*
_output_shapes
: *
dtype0*0
value'B% Bqangle_weights_weights_conv2d_3
Е
qangle_weights_weights_conv2d_3ScalarSummary$qangle_weights_weights_conv2d_3/tagsTensordot_17*
T0*
_output_shapes
: 
Д
$qangle_weights_weights_conv2d_4/tagsConst*
_output_shapes
: *
dtype0*0
value'B% Bqangle_weights_weights_conv2d_4
Е
qangle_weights_weights_conv2d_4ScalarSummary$qangle_weights_weights_conv2d_4/tagsTensordot_19*
T0*
_output_shapes
: 
r
qangle_weights_average/tagsConst*
_output_shapes
: *
dtype0*'
valueB Bqangle_weights_average
m
qangle_weights_averageScalarSummaryqangle_weights_average/tagsMean_8*
T0*
_output_shapes
: 
У

initNoOp^beta1_power/Assign^beta2_power/Assign^conv2d/kernel/Adam/Assign^conv2d/kernel/Adam_1/Assign^conv2d/kernel/Assign^conv2d_1/kernel/Adam/Assign^conv2d_1/kernel/Adam_1/Assign^conv2d_1/kernel/Assign^conv2d_2/kernel/Adam/Assign^conv2d_2/kernel/Adam_1/Assign^conv2d_2/kernel/Assign^conv2d_3/kernel/Adam/Assign^conv2d_3/kernel/Adam_1/Assign^conv2d_3/kernel/Assign^conv2d_4/kernel/Adam/Assign^conv2d_4/kernel/Adam_1/Assign^conv2d_4/kernel/Assign^intercept/Adam/Assign^intercept/Adam_1/Assign^intercept/Assign^intercept_1/Adam/Assign^intercept_1/Adam_1/Assign^intercept_1/Assign^intercept_2/Adam/Assign^intercept_2/Adam_1/Assign^intercept_2/Assign^intercept_3/Adam/Assign^intercept_3/Adam_1/Assign^intercept_3/Assign^intercept_4/Adam/Assign^intercept_4/Adam_1/Assign^intercept_4/Assign^intercept_5/Adam/Assign^intercept_5/Adam_1/Assign^intercept_5/Assign^slope/Adam/Assign^slope/Adam_1/Assign^slope/Assign^slope_1/Adam/Assign^slope_1/Adam_1/Assign^slope_1/Assign^slope_2/Adam/Assign^slope_2/Adam_1/Assign^slope_2/Assign^slope_3/Adam/Assign^slope_3/Adam_1/Assign^slope_3/Assign^slope_4/Adam/Assign^slope_4/Adam_1/Assign^slope_4/Assign^slope_5/Adam/Assign^slope_5/Adam_1/Assign^slope_5/Assign
Н	
Merge/MergeSummaryMergeSummaryhp_cross_entropy_flatten_1distillation_loss_flatten_1b_i_t___l_o_s_s%r_e_g_u_l_a_r_i_z_a_t_i_o_n___t_e_r_mlp_accuracy_flatten_1hp_accuracy_flatten_1total_loss_total_lossweights_bits_conv2dweights_bits_conv2d_1weights_bits_conv2d_2weights_bits_activation_3weights_bits_conv2d_3weights_bits_conv2d_4bits_averageqerr_op_activationqerr_op_activation_1qerr_op_activation_2qerr_op_activation_3qerr_op_activation_4qerr_op_averageqerr_weights_bias_conv2dqerr_weights_bias_conv2d_1qerr_weights_bias_conv2d_2qerr_weights_bias_conv2d_3qerr_weights_bias_conv2d_4qerr_weights_weights_conv2dqerr_weights_weights_conv2d_1qerr_weights_weights_conv2d_2qerr_weights_weights_conv2d_3qerr_weights_weights_conv2d_4qerr_weights_averageqangle_op_activationqangle_op_activation_1qangle_op_activation_2qangle_op_activation_3qangle_op_activation_4qangle_op_averageqangle_weights_weights_conv2dqangle_weights_weights_conv2d_1qangle_weights_weights_conv2d_2qangle_weights_weights_conv2d_3qangle_weights_weights_conv2d_4qangle_weights_average*
N+*
_output_shapes
: 
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
Д
save/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a5fdef5854ed4e8781c65041888ab67b/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
\
save/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
┴
save/SaveV2/tensor_namesConst*
_output_shapes
:5*
dtype0*Ї
valueъBч5Bbeta1_powerBbeta2_powerBconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1Bconv2d_2/kernelBconv2d_2/kernel/AdamBconv2d_2/kernel/Adam_1Bconv2d_3/kernelBconv2d_3/kernel/AdamBconv2d_3/kernel/Adam_1Bconv2d_4/kernelBconv2d_4/kernel/AdamBconv2d_4/kernel/Adam_1B	interceptBintercept/AdamBintercept/Adam_1Bintercept_1Bintercept_1/AdamBintercept_1/Adam_1Bintercept_2Bintercept_2/AdamBintercept_2/Adam_1Bintercept_3Bintercept_3/AdamBintercept_3/Adam_1Bintercept_4Bintercept_4/AdamBintercept_4/Adam_1Bintercept_5Bintercept_5/AdamBintercept_5/Adam_1BslopeB
slope/AdamBslope/Adam_1Bslope_1Bslope_1/AdamBslope_1/Adam_1Bslope_2Bslope_2/AdamBslope_2/Adam_1Bslope_3Bslope_3/AdamBslope_3/Adam_1Bslope_4Bslope_4/AdamBslope_4/Adam_1Bslope_5Bslope_5/AdamBslope_5/Adam_1
═
save/SaveV2/shape_and_slicesConst*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
л
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_power/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOp&conv2d/kernel/Adam/Read/ReadVariableOp(conv2d/kernel/Adam_1/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp(conv2d_1/kernel/Adam/Read/ReadVariableOp*conv2d_1/kernel/Adam_1/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp(conv2d_2/kernel/Adam/Read/ReadVariableOp*conv2d_2/kernel/Adam_1/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp(conv2d_3/kernel/Adam/Read/ReadVariableOp*conv2d_3/kernel/Adam_1/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp(conv2d_4/kernel/Adam/Read/ReadVariableOp*conv2d_4/kernel/Adam_1/Read/ReadVariableOpintercept/Read/ReadVariableOp"intercept/Adam/Read/ReadVariableOp$intercept/Adam_1/Read/ReadVariableOpintercept_1/Read/ReadVariableOp$intercept_1/Adam/Read/ReadVariableOp&intercept_1/Adam_1/Read/ReadVariableOpintercept_2/Read/ReadVariableOp$intercept_2/Adam/Read/ReadVariableOp&intercept_2/Adam_1/Read/ReadVariableOpintercept_3/Read/ReadVariableOp$intercept_3/Adam/Read/ReadVariableOp&intercept_3/Adam_1/Read/ReadVariableOpintercept_4/Read/ReadVariableOp$intercept_4/Adam/Read/ReadVariableOp&intercept_4/Adam_1/Read/ReadVariableOpintercept_5/Read/ReadVariableOp$intercept_5/Adam/Read/ReadVariableOp&intercept_5/Adam_1/Read/ReadVariableOpslope/Read/ReadVariableOpslope/Adam/Read/ReadVariableOp slope/Adam_1/Read/ReadVariableOpslope_1/Read/ReadVariableOp slope_1/Adam/Read/ReadVariableOp"slope_1/Adam_1/Read/ReadVariableOpslope_2/Read/ReadVariableOp slope_2/Adam/Read/ReadVariableOp"slope_2/Adam_1/Read/ReadVariableOpslope_3/Read/ReadVariableOp slope_3/Adam/Read/ReadVariableOp"slope_3/Adam_1/Read/ReadVariableOpslope_4/Read/ReadVariableOp slope_4/Adam/Read/ReadVariableOp"slope_4/Adam_1/Read/ReadVariableOpslope_5/Read/ReadVariableOp slope_5/Adam/Read/ReadVariableOp"slope_5/Adam_1/Read/ReadVariableOp*C
dtypes9
725
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
T0*
_output_shapes
:*

axis 
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
─
save/RestoreV2/tensor_namesConst*
_output_shapes
:5*
dtype0*Ї
valueъBч5Bbeta1_powerBbeta2_powerBconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1Bconv2d_2/kernelBconv2d_2/kernel/AdamBconv2d_2/kernel/Adam_1Bconv2d_3/kernelBconv2d_3/kernel/AdamBconv2d_3/kernel/Adam_1Bconv2d_4/kernelBconv2d_4/kernel/AdamBconv2d_4/kernel/Adam_1B	interceptBintercept/AdamBintercept/Adam_1Bintercept_1Bintercept_1/AdamBintercept_1/Adam_1Bintercept_2Bintercept_2/AdamBintercept_2/Adam_1Bintercept_3Bintercept_3/AdamBintercept_3/Adam_1Bintercept_4Bintercept_4/AdamBintercept_4/Adam_1Bintercept_5Bintercept_5/AdamBintercept_5/Adam_1BslopeB
slope/AdamBslope/Adam_1Bslope_1Bslope_1/AdamBslope_1/Adam_1Bslope_2Bslope_2/AdamBslope_2/Adam_1Bslope_3Bslope_3/AdamBslope_3/Adam_1Bslope_4Bslope_4/AdamBslope_4/Adam_1Bslope_5Bslope_5/AdamBslope_5/Adam_1
╨
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ч
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*ъ
_output_shapes╫
╘:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
T
save/AssignVariableOpAssignVariableOpbeta1_powersave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
V
save/AssignVariableOp_1AssignVariableOpbeta2_powersave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
X
save/AssignVariableOp_2AssignVariableOpconv2d/kernelsave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
]
save/AssignVariableOp_3AssignVariableOpconv2d/kernel/Adamsave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
_
save/AssignVariableOp_4AssignVariableOpconv2d/kernel/Adam_1save/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
Z
save/AssignVariableOp_5AssignVariableOpconv2d_1/kernelsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
_
save/AssignVariableOp_6AssignVariableOpconv2d_1/kernel/Adamsave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
a
save/AssignVariableOp_7AssignVariableOpconv2d_1/kernel/Adam_1save/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
Z
save/AssignVariableOp_8AssignVariableOpconv2d_2/kernelsave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
`
save/AssignVariableOp_9AssignVariableOpconv2d_2/kernel/Adamsave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
c
save/AssignVariableOp_10AssignVariableOpconv2d_2/kernel/Adam_1save/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
\
save/AssignVariableOp_11AssignVariableOpconv2d_3/kernelsave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0*
_output_shapes
:
a
save/AssignVariableOp_12AssignVariableOpconv2d_3/kernel/Adamsave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
c
save/AssignVariableOp_13AssignVariableOpconv2d_3/kernel/Adam_1save/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
T0*
_output_shapes
:
\
save/AssignVariableOp_14AssignVariableOpconv2d_4/kernelsave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
a
save/AssignVariableOp_15AssignVariableOpconv2d_4/kernel/Adamsave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
T0*
_output_shapes
:
c
save/AssignVariableOp_16AssignVariableOpconv2d_4/kernel/Adam_1save/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
T0*
_output_shapes
:
V
save/AssignVariableOp_17AssignVariableOp	interceptsave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
[
save/AssignVariableOp_18AssignVariableOpintercept/Adamsave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
T0*
_output_shapes
:
]
save/AssignVariableOp_19AssignVariableOpintercept/Adam_1save/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
T0*
_output_shapes
:
X
save/AssignVariableOp_20AssignVariableOpintercept_1save/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
T0*
_output_shapes
:
]
save/AssignVariableOp_21AssignVariableOpintercept_1/Adamsave/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:
_
save/AssignVariableOp_22AssignVariableOpintercept_1/Adam_1save/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
T0*
_output_shapes
:
X
save/AssignVariableOp_23AssignVariableOpintercept_2save/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
T0*
_output_shapes
:
]
save/AssignVariableOp_24AssignVariableOpintercept_2/Adamsave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
T0*
_output_shapes
:
_
save/AssignVariableOp_25AssignVariableOpintercept_2/Adam_1save/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:26*
T0*
_output_shapes
:
X
save/AssignVariableOp_26AssignVariableOpintercept_3save/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:27*
T0*
_output_shapes
:
]
save/AssignVariableOp_27AssignVariableOpintercept_3/Adamsave/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:28*
T0*
_output_shapes
:
_
save/AssignVariableOp_28AssignVariableOpintercept_3/Adam_1save/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:29*
T0*
_output_shapes
:
X
save/AssignVariableOp_29AssignVariableOpintercept_4save/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:30*
T0*
_output_shapes
:
]
save/AssignVariableOp_30AssignVariableOpintercept_4/Adamsave/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:31*
T0*
_output_shapes
:
_
save/AssignVariableOp_31AssignVariableOpintercept_4/Adam_1save/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:32*
T0*
_output_shapes
:
X
save/AssignVariableOp_32AssignVariableOpintercept_5save/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:33*
T0*
_output_shapes
:
]
save/AssignVariableOp_33AssignVariableOpintercept_5/Adamsave/Identity_34*
dtype0
R
save/Identity_35Identitysave/RestoreV2:34*
T0*
_output_shapes
:
_
save/AssignVariableOp_34AssignVariableOpintercept_5/Adam_1save/Identity_35*
dtype0
R
save/Identity_36Identitysave/RestoreV2:35*
T0*
_output_shapes
:
R
save/AssignVariableOp_35AssignVariableOpslopesave/Identity_36*
dtype0
R
save/Identity_37Identitysave/RestoreV2:36*
T0*
_output_shapes
:
W
save/AssignVariableOp_36AssignVariableOp
slope/Adamsave/Identity_37*
dtype0
R
save/Identity_38Identitysave/RestoreV2:37*
T0*
_output_shapes
:
Y
save/AssignVariableOp_37AssignVariableOpslope/Adam_1save/Identity_38*
dtype0
R
save/Identity_39Identitysave/RestoreV2:38*
T0*
_output_shapes
:
T
save/AssignVariableOp_38AssignVariableOpslope_1save/Identity_39*
dtype0
R
save/Identity_40Identitysave/RestoreV2:39*
T0*
_output_shapes
:
Y
save/AssignVariableOp_39AssignVariableOpslope_1/Adamsave/Identity_40*
dtype0
R
save/Identity_41Identitysave/RestoreV2:40*
T0*
_output_shapes
:
[
save/AssignVariableOp_40AssignVariableOpslope_1/Adam_1save/Identity_41*
dtype0
R
save/Identity_42Identitysave/RestoreV2:41*
T0*
_output_shapes
:
T
save/AssignVariableOp_41AssignVariableOpslope_2save/Identity_42*
dtype0
R
save/Identity_43Identitysave/RestoreV2:42*
T0*
_output_shapes
:
Y
save/AssignVariableOp_42AssignVariableOpslope_2/Adamsave/Identity_43*
dtype0
R
save/Identity_44Identitysave/RestoreV2:43*
T0*
_output_shapes
:
[
save/AssignVariableOp_43AssignVariableOpslope_2/Adam_1save/Identity_44*
dtype0
R
save/Identity_45Identitysave/RestoreV2:44*
T0*
_output_shapes
:
T
save/AssignVariableOp_44AssignVariableOpslope_3save/Identity_45*
dtype0
R
save/Identity_46Identitysave/RestoreV2:45*
T0*
_output_shapes
:
Y
save/AssignVariableOp_45AssignVariableOpslope_3/Adamsave/Identity_46*
dtype0
R
save/Identity_47Identitysave/RestoreV2:46*
T0*
_output_shapes
:
[
save/AssignVariableOp_46AssignVariableOpslope_3/Adam_1save/Identity_47*
dtype0
R
save/Identity_48Identitysave/RestoreV2:47*
T0*
_output_shapes
:
T
save/AssignVariableOp_47AssignVariableOpslope_4save/Identity_48*
dtype0
R
save/Identity_49Identitysave/RestoreV2:48*
T0*
_output_shapes
:
Y
save/AssignVariableOp_48AssignVariableOpslope_4/Adamsave/Identity_49*
dtype0
R
save/Identity_50Identitysave/RestoreV2:49*
T0*
_output_shapes
:
[
save/AssignVariableOp_49AssignVariableOpslope_4/Adam_1save/Identity_50*
dtype0
R
save/Identity_51Identitysave/RestoreV2:50*
T0*
_output_shapes
:
T
save/AssignVariableOp_50AssignVariableOpslope_5save/Identity_51*
dtype0
R
save/Identity_52Identitysave/RestoreV2:51*
T0*
_output_shapes
:
Y
save/AssignVariableOp_51AssignVariableOpslope_5/Adamsave/Identity_52*
dtype0
R
save/Identity_53Identitysave/RestoreV2:52*
T0*
_output_shapes
:
[
save/AssignVariableOp_52AssignVariableOpslope_5/Adam_1save/Identity_53*
dtype0
е
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_39^save/AssignVariableOp_4^save/AssignVariableOp_40^save/AssignVariableOp_41^save/AssignVariableOp_42^save/AssignVariableOp_43^save/AssignVariableOp_44^save/AssignVariableOp_45^save/AssignVariableOp_46^save/AssignVariableOp_47^save/AssignVariableOp_48^save/AssignVariableOp_49^save/AssignVariableOp_5^save/AssignVariableOp_50^save/AssignVariableOp_51^save/AssignVariableOp_52^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
dtype0*
shape: 
Ж
save_1/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_8e2431b1b3af49beaa53138ca85f11ca/part
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_1/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
Е
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
├
save_1/SaveV2/tensor_namesConst*
_output_shapes
:5*
dtype0*Ї
valueъBч5Bbeta1_powerBbeta2_powerBconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1Bconv2d_2/kernelBconv2d_2/kernel/AdamBconv2d_2/kernel/Adam_1Bconv2d_3/kernelBconv2d_3/kernel/AdamBconv2d_3/kernel/Adam_1Bconv2d_4/kernelBconv2d_4/kernel/AdamBconv2d_4/kernel/Adam_1B	interceptBintercept/AdamBintercept/Adam_1Bintercept_1Bintercept_1/AdamBintercept_1/Adam_1Bintercept_2Bintercept_2/AdamBintercept_2/Adam_1Bintercept_3Bintercept_3/AdamBintercept_3/Adam_1Bintercept_4Bintercept_4/AdamBintercept_4/Adam_1Bintercept_5Bintercept_5/AdamBintercept_5/Adam_1BslopeB
slope/AdamBslope/Adam_1Bslope_1Bslope_1/AdamBslope_1/Adam_1Bslope_2Bslope_2/AdamBslope_2/Adam_1Bslope_3Bslope_3/AdamBslope_3/Adam_1Bslope_4Bslope_4/AdamBslope_4/Adam_1Bslope_5Bslope_5/AdamBslope_5/Adam_1
╧
save_1/SaveV2/shape_and_slicesConst*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
│
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_power/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOp&conv2d/kernel/Adam/Read/ReadVariableOp(conv2d/kernel/Adam_1/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp(conv2d_1/kernel/Adam/Read/ReadVariableOp*conv2d_1/kernel/Adam_1/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp(conv2d_2/kernel/Adam/Read/ReadVariableOp*conv2d_2/kernel/Adam_1/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp(conv2d_3/kernel/Adam/Read/ReadVariableOp*conv2d_3/kernel/Adam_1/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp(conv2d_4/kernel/Adam/Read/ReadVariableOp*conv2d_4/kernel/Adam_1/Read/ReadVariableOpintercept/Read/ReadVariableOp"intercept/Adam/Read/ReadVariableOp$intercept/Adam_1/Read/ReadVariableOpintercept_1/Read/ReadVariableOp$intercept_1/Adam/Read/ReadVariableOp&intercept_1/Adam_1/Read/ReadVariableOpintercept_2/Read/ReadVariableOp$intercept_2/Adam/Read/ReadVariableOp&intercept_2/Adam_1/Read/ReadVariableOpintercept_3/Read/ReadVariableOp$intercept_3/Adam/Read/ReadVariableOp&intercept_3/Adam_1/Read/ReadVariableOpintercept_4/Read/ReadVariableOp$intercept_4/Adam/Read/ReadVariableOp&intercept_4/Adam_1/Read/ReadVariableOpintercept_5/Read/ReadVariableOp$intercept_5/Adam/Read/ReadVariableOp&intercept_5/Adam_1/Read/ReadVariableOpslope/Read/ReadVariableOpslope/Adam/Read/ReadVariableOp slope/Adam_1/Read/ReadVariableOpslope_1/Read/ReadVariableOp slope_1/Adam/Read/ReadVariableOp"slope_1/Adam_1/Read/ReadVariableOpslope_2/Read/ReadVariableOp slope_2/Adam/Read/ReadVariableOp"slope_2/Adam_1/Read/ReadVariableOpslope_3/Read/ReadVariableOp slope_3/Adam/Read/ReadVariableOp"slope_3/Adam_1/Read/ReadVariableOpslope_4/Read/ReadVariableOp slope_4/Adam/Read/ReadVariableOp"slope_4/Adam_1/Read/ReadVariableOpslope_5/Read/ReadVariableOp slope_5/Adam/Read/ReadVariableOp"slope_5/Adam_1/Read/ReadVariableOp*C
dtypes9
725
Щ
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
г
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
N*
T0*
_output_shapes
:*

axis 
Г
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
В
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
╞
save_1/RestoreV2/tensor_namesConst*
_output_shapes
:5*
dtype0*Ї
valueъBч5Bbeta1_powerBbeta2_powerBconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1Bconv2d_2/kernelBconv2d_2/kernel/AdamBconv2d_2/kernel/Adam_1Bconv2d_3/kernelBconv2d_3/kernel/AdamBconv2d_3/kernel/Adam_1Bconv2d_4/kernelBconv2d_4/kernel/AdamBconv2d_4/kernel/Adam_1B	interceptBintercept/AdamBintercept/Adam_1Bintercept_1Bintercept_1/AdamBintercept_1/Adam_1Bintercept_2Bintercept_2/AdamBintercept_2/Adam_1Bintercept_3Bintercept_3/AdamBintercept_3/Adam_1Bintercept_4Bintercept_4/AdamBintercept_4/Adam_1Bintercept_5Bintercept_5/AdamBintercept_5/Adam_1BslopeB
slope/AdamBslope/Adam_1Bslope_1Bslope_1/AdamBslope_1/Adam_1Bslope_2Bslope_2/AdamBslope_2/Adam_1Bslope_3Bslope_3/AdamBslope_3/Adam_1Bslope_4Bslope_4/AdamBslope_4/Adam_1Bslope_5Bslope_5/AdamBslope_5/Adam_1
╥
!save_1/RestoreV2/shape_and_slicesConst*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Я
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*ъ
_output_shapes╫
╘:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725
R
save_1/Identity_1Identitysave_1/RestoreV2*
T0*
_output_shapes
:
X
save_1/AssignVariableOpAssignVariableOpbeta1_powersave_1/Identity_1*
dtype0
T
save_1/Identity_2Identitysave_1/RestoreV2:1*
T0*
_output_shapes
:
Z
save_1/AssignVariableOp_1AssignVariableOpbeta2_powersave_1/Identity_2*
dtype0
T
save_1/Identity_3Identitysave_1/RestoreV2:2*
T0*
_output_shapes
:
\
save_1/AssignVariableOp_2AssignVariableOpconv2d/kernelsave_1/Identity_3*
dtype0
T
save_1/Identity_4Identitysave_1/RestoreV2:3*
T0*
_output_shapes
:
a
save_1/AssignVariableOp_3AssignVariableOpconv2d/kernel/Adamsave_1/Identity_4*
dtype0
T
save_1/Identity_5Identitysave_1/RestoreV2:4*
T0*
_output_shapes
:
c
save_1/AssignVariableOp_4AssignVariableOpconv2d/kernel/Adam_1save_1/Identity_5*
dtype0
T
save_1/Identity_6Identitysave_1/RestoreV2:5*
T0*
_output_shapes
:
^
save_1/AssignVariableOp_5AssignVariableOpconv2d_1/kernelsave_1/Identity_6*
dtype0
T
save_1/Identity_7Identitysave_1/RestoreV2:6*
T0*
_output_shapes
:
c
save_1/AssignVariableOp_6AssignVariableOpconv2d_1/kernel/Adamsave_1/Identity_7*
dtype0
T
save_1/Identity_8Identitysave_1/RestoreV2:7*
T0*
_output_shapes
:
e
save_1/AssignVariableOp_7AssignVariableOpconv2d_1/kernel/Adam_1save_1/Identity_8*
dtype0
T
save_1/Identity_9Identitysave_1/RestoreV2:8*
T0*
_output_shapes
:
^
save_1/AssignVariableOp_8AssignVariableOpconv2d_2/kernelsave_1/Identity_9*
dtype0
U
save_1/Identity_10Identitysave_1/RestoreV2:9*
T0*
_output_shapes
:
d
save_1/AssignVariableOp_9AssignVariableOpconv2d_2/kernel/Adamsave_1/Identity_10*
dtype0
V
save_1/Identity_11Identitysave_1/RestoreV2:10*
T0*
_output_shapes
:
g
save_1/AssignVariableOp_10AssignVariableOpconv2d_2/kernel/Adam_1save_1/Identity_11*
dtype0
V
save_1/Identity_12Identitysave_1/RestoreV2:11*
T0*
_output_shapes
:
`
save_1/AssignVariableOp_11AssignVariableOpconv2d_3/kernelsave_1/Identity_12*
dtype0
V
save_1/Identity_13Identitysave_1/RestoreV2:12*
T0*
_output_shapes
:
e
save_1/AssignVariableOp_12AssignVariableOpconv2d_3/kernel/Adamsave_1/Identity_13*
dtype0
V
save_1/Identity_14Identitysave_1/RestoreV2:13*
T0*
_output_shapes
:
g
save_1/AssignVariableOp_13AssignVariableOpconv2d_3/kernel/Adam_1save_1/Identity_14*
dtype0
V
save_1/Identity_15Identitysave_1/RestoreV2:14*
T0*
_output_shapes
:
`
save_1/AssignVariableOp_14AssignVariableOpconv2d_4/kernelsave_1/Identity_15*
dtype0
V
save_1/Identity_16Identitysave_1/RestoreV2:15*
T0*
_output_shapes
:
e
save_1/AssignVariableOp_15AssignVariableOpconv2d_4/kernel/Adamsave_1/Identity_16*
dtype0
V
save_1/Identity_17Identitysave_1/RestoreV2:16*
T0*
_output_shapes
:
g
save_1/AssignVariableOp_16AssignVariableOpconv2d_4/kernel/Adam_1save_1/Identity_17*
dtype0
V
save_1/Identity_18Identitysave_1/RestoreV2:17*
T0*
_output_shapes
:
Z
save_1/AssignVariableOp_17AssignVariableOp	interceptsave_1/Identity_18*
dtype0
V
save_1/Identity_19Identitysave_1/RestoreV2:18*
T0*
_output_shapes
:
_
save_1/AssignVariableOp_18AssignVariableOpintercept/Adamsave_1/Identity_19*
dtype0
V
save_1/Identity_20Identitysave_1/RestoreV2:19*
T0*
_output_shapes
:
a
save_1/AssignVariableOp_19AssignVariableOpintercept/Adam_1save_1/Identity_20*
dtype0
V
save_1/Identity_21Identitysave_1/RestoreV2:20*
T0*
_output_shapes
:
\
save_1/AssignVariableOp_20AssignVariableOpintercept_1save_1/Identity_21*
dtype0
V
save_1/Identity_22Identitysave_1/RestoreV2:21*
T0*
_output_shapes
:
a
save_1/AssignVariableOp_21AssignVariableOpintercept_1/Adamsave_1/Identity_22*
dtype0
V
save_1/Identity_23Identitysave_1/RestoreV2:22*
T0*
_output_shapes
:
c
save_1/AssignVariableOp_22AssignVariableOpintercept_1/Adam_1save_1/Identity_23*
dtype0
V
save_1/Identity_24Identitysave_1/RestoreV2:23*
T0*
_output_shapes
:
\
save_1/AssignVariableOp_23AssignVariableOpintercept_2save_1/Identity_24*
dtype0
V
save_1/Identity_25Identitysave_1/RestoreV2:24*
T0*
_output_shapes
:
a
save_1/AssignVariableOp_24AssignVariableOpintercept_2/Adamsave_1/Identity_25*
dtype0
V
save_1/Identity_26Identitysave_1/RestoreV2:25*
T0*
_output_shapes
:
c
save_1/AssignVariableOp_25AssignVariableOpintercept_2/Adam_1save_1/Identity_26*
dtype0
V
save_1/Identity_27Identitysave_1/RestoreV2:26*
T0*
_output_shapes
:
\
save_1/AssignVariableOp_26AssignVariableOpintercept_3save_1/Identity_27*
dtype0
V
save_1/Identity_28Identitysave_1/RestoreV2:27*
T0*
_output_shapes
:
a
save_1/AssignVariableOp_27AssignVariableOpintercept_3/Adamsave_1/Identity_28*
dtype0
V
save_1/Identity_29Identitysave_1/RestoreV2:28*
T0*
_output_shapes
:
c
save_1/AssignVariableOp_28AssignVariableOpintercept_3/Adam_1save_1/Identity_29*
dtype0
V
save_1/Identity_30Identitysave_1/RestoreV2:29*
T0*
_output_shapes
:
\
save_1/AssignVariableOp_29AssignVariableOpintercept_4save_1/Identity_30*
dtype0
V
save_1/Identity_31Identitysave_1/RestoreV2:30*
T0*
_output_shapes
:
a
save_1/AssignVariableOp_30AssignVariableOpintercept_4/Adamsave_1/Identity_31*
dtype0
V
save_1/Identity_32Identitysave_1/RestoreV2:31*
T0*
_output_shapes
:
c
save_1/AssignVariableOp_31AssignVariableOpintercept_4/Adam_1save_1/Identity_32*
dtype0
V
save_1/Identity_33Identitysave_1/RestoreV2:32*
T0*
_output_shapes
:
\
save_1/AssignVariableOp_32AssignVariableOpintercept_5save_1/Identity_33*
dtype0
V
save_1/Identity_34Identitysave_1/RestoreV2:33*
T0*
_output_shapes
:
a
save_1/AssignVariableOp_33AssignVariableOpintercept_5/Adamsave_1/Identity_34*
dtype0
V
save_1/Identity_35Identitysave_1/RestoreV2:34*
T0*
_output_shapes
:
c
save_1/AssignVariableOp_34AssignVariableOpintercept_5/Adam_1save_1/Identity_35*
dtype0
V
save_1/Identity_36Identitysave_1/RestoreV2:35*
T0*
_output_shapes
:
V
save_1/AssignVariableOp_35AssignVariableOpslopesave_1/Identity_36*
dtype0
V
save_1/Identity_37Identitysave_1/RestoreV2:36*
T0*
_output_shapes
:
[
save_1/AssignVariableOp_36AssignVariableOp
slope/Adamsave_1/Identity_37*
dtype0
V
save_1/Identity_38Identitysave_1/RestoreV2:37*
T0*
_output_shapes
:
]
save_1/AssignVariableOp_37AssignVariableOpslope/Adam_1save_1/Identity_38*
dtype0
V
save_1/Identity_39Identitysave_1/RestoreV2:38*
T0*
_output_shapes
:
X
save_1/AssignVariableOp_38AssignVariableOpslope_1save_1/Identity_39*
dtype0
V
save_1/Identity_40Identitysave_1/RestoreV2:39*
T0*
_output_shapes
:
]
save_1/AssignVariableOp_39AssignVariableOpslope_1/Adamsave_1/Identity_40*
dtype0
V
save_1/Identity_41Identitysave_1/RestoreV2:40*
T0*
_output_shapes
:
_
save_1/AssignVariableOp_40AssignVariableOpslope_1/Adam_1save_1/Identity_41*
dtype0
V
save_1/Identity_42Identitysave_1/RestoreV2:41*
T0*
_output_shapes
:
X
save_1/AssignVariableOp_41AssignVariableOpslope_2save_1/Identity_42*
dtype0
V
save_1/Identity_43Identitysave_1/RestoreV2:42*
T0*
_output_shapes
:
]
save_1/AssignVariableOp_42AssignVariableOpslope_2/Adamsave_1/Identity_43*
dtype0
V
save_1/Identity_44Identitysave_1/RestoreV2:43*
T0*
_output_shapes
:
_
save_1/AssignVariableOp_43AssignVariableOpslope_2/Adam_1save_1/Identity_44*
dtype0
V
save_1/Identity_45Identitysave_1/RestoreV2:44*
T0*
_output_shapes
:
X
save_1/AssignVariableOp_44AssignVariableOpslope_3save_1/Identity_45*
dtype0
V
save_1/Identity_46Identitysave_1/RestoreV2:45*
T0*
_output_shapes
:
]
save_1/AssignVariableOp_45AssignVariableOpslope_3/Adamsave_1/Identity_46*
dtype0
V
save_1/Identity_47Identitysave_1/RestoreV2:46*
T0*
_output_shapes
:
_
save_1/AssignVariableOp_46AssignVariableOpslope_3/Adam_1save_1/Identity_47*
dtype0
V
save_1/Identity_48Identitysave_1/RestoreV2:47*
T0*
_output_shapes
:
X
save_1/AssignVariableOp_47AssignVariableOpslope_4save_1/Identity_48*
dtype0
V
save_1/Identity_49Identitysave_1/RestoreV2:48*
T0*
_output_shapes
:
]
save_1/AssignVariableOp_48AssignVariableOpslope_4/Adamsave_1/Identity_49*
dtype0
V
save_1/Identity_50Identitysave_1/RestoreV2:49*
T0*
_output_shapes
:
_
save_1/AssignVariableOp_49AssignVariableOpslope_4/Adam_1save_1/Identity_50*
dtype0
V
save_1/Identity_51Identitysave_1/RestoreV2:50*
T0*
_output_shapes
:
X
save_1/AssignVariableOp_50AssignVariableOpslope_5save_1/Identity_51*
dtype0
V
save_1/Identity_52Identitysave_1/RestoreV2:51*
T0*
_output_shapes
:
]
save_1/AssignVariableOp_51AssignVariableOpslope_5/Adamsave_1/Identity_52*
dtype0
V
save_1/Identity_53Identitysave_1/RestoreV2:52*
T0*
_output_shapes
:
_
save_1/AssignVariableOp_52AssignVariableOpslope_5/Adam_1save_1/Identity_53*
dtype0
С
save_1/restore_shardNoOp^save_1/AssignVariableOp^save_1/AssignVariableOp_1^save_1/AssignVariableOp_10^save_1/AssignVariableOp_11^save_1/AssignVariableOp_12^save_1/AssignVariableOp_13^save_1/AssignVariableOp_14^save_1/AssignVariableOp_15^save_1/AssignVariableOp_16^save_1/AssignVariableOp_17^save_1/AssignVariableOp_18^save_1/AssignVariableOp_19^save_1/AssignVariableOp_2^save_1/AssignVariableOp_20^save_1/AssignVariableOp_21^save_1/AssignVariableOp_22^save_1/AssignVariableOp_23^save_1/AssignVariableOp_24^save_1/AssignVariableOp_25^save_1/AssignVariableOp_26^save_1/AssignVariableOp_27^save_1/AssignVariableOp_28^save_1/AssignVariableOp_29^save_1/AssignVariableOp_3^save_1/AssignVariableOp_30^save_1/AssignVariableOp_31^save_1/AssignVariableOp_32^save_1/AssignVariableOp_33^save_1/AssignVariableOp_34^save_1/AssignVariableOp_35^save_1/AssignVariableOp_36^save_1/AssignVariableOp_37^save_1/AssignVariableOp_38^save_1/AssignVariableOp_39^save_1/AssignVariableOp_4^save_1/AssignVariableOp_40^save_1/AssignVariableOp_41^save_1/AssignVariableOp_42^save_1/AssignVariableOp_43^save_1/AssignVariableOp_44^save_1/AssignVariableOp_45^save_1/AssignVariableOp_46^save_1/AssignVariableOp_47^save_1/AssignVariableOp_48^save_1/AssignVariableOp_49^save_1/AssignVariableOp_5^save_1/AssignVariableOp_50^save_1/AssignVariableOp_51^save_1/AssignVariableOp_52^save_1/AssignVariableOp_6^save_1/AssignVariableOp_7^save_1/AssignVariableOp_8^save_1/AssignVariableOp_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
_output_shapes
: *
dtype0*
shape: 
Ж
save_2/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_14d5bfd7dae84d1c972709d9d3e0b8bc/part
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_2/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_2/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
Е
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
├
save_2/SaveV2/tensor_namesConst*
_output_shapes
:5*
dtype0*Ї
valueъBч5Bbeta1_powerBbeta2_powerBconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1Bconv2d_2/kernelBconv2d_2/kernel/AdamBconv2d_2/kernel/Adam_1Bconv2d_3/kernelBconv2d_3/kernel/AdamBconv2d_3/kernel/Adam_1Bconv2d_4/kernelBconv2d_4/kernel/AdamBconv2d_4/kernel/Adam_1B	interceptBintercept/AdamBintercept/Adam_1Bintercept_1Bintercept_1/AdamBintercept_1/Adam_1Bintercept_2Bintercept_2/AdamBintercept_2/Adam_1Bintercept_3Bintercept_3/AdamBintercept_3/Adam_1Bintercept_4Bintercept_4/AdamBintercept_4/Adam_1Bintercept_5Bintercept_5/AdamBintercept_5/Adam_1BslopeB
slope/AdamBslope/Adam_1Bslope_1Bslope_1/AdamBslope_1/Adam_1Bslope_2Bslope_2/AdamBslope_2/Adam_1Bslope_3Bslope_3/AdamBslope_3/Adam_1Bslope_4Bslope_4/AdamBslope_4/Adam_1Bslope_5Bslope_5/AdamBslope_5/Adam_1
╧
save_2/SaveV2/shape_and_slicesConst*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
│
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_power/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOp&conv2d/kernel/Adam/Read/ReadVariableOp(conv2d/kernel/Adam_1/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp(conv2d_1/kernel/Adam/Read/ReadVariableOp*conv2d_1/kernel/Adam_1/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp(conv2d_2/kernel/Adam/Read/ReadVariableOp*conv2d_2/kernel/Adam_1/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp(conv2d_3/kernel/Adam/Read/ReadVariableOp*conv2d_3/kernel/Adam_1/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp(conv2d_4/kernel/Adam/Read/ReadVariableOp*conv2d_4/kernel/Adam_1/Read/ReadVariableOpintercept/Read/ReadVariableOp"intercept/Adam/Read/ReadVariableOp$intercept/Adam_1/Read/ReadVariableOpintercept_1/Read/ReadVariableOp$intercept_1/Adam/Read/ReadVariableOp&intercept_1/Adam_1/Read/ReadVariableOpintercept_2/Read/ReadVariableOp$intercept_2/Adam/Read/ReadVariableOp&intercept_2/Adam_1/Read/ReadVariableOpintercept_3/Read/ReadVariableOp$intercept_3/Adam/Read/ReadVariableOp&intercept_3/Adam_1/Read/ReadVariableOpintercept_4/Read/ReadVariableOp$intercept_4/Adam/Read/ReadVariableOp&intercept_4/Adam_1/Read/ReadVariableOpintercept_5/Read/ReadVariableOp$intercept_5/Adam/Read/ReadVariableOp&intercept_5/Adam_1/Read/ReadVariableOpslope/Read/ReadVariableOpslope/Adam/Read/ReadVariableOp slope/Adam_1/Read/ReadVariableOpslope_1/Read/ReadVariableOp slope_1/Adam/Read/ReadVariableOp"slope_1/Adam_1/Read/ReadVariableOpslope_2/Read/ReadVariableOp slope_2/Adam/Read/ReadVariableOp"slope_2/Adam_1/Read/ReadVariableOpslope_3/Read/ReadVariableOp slope_3/Adam/Read/ReadVariableOp"slope_3/Adam_1/Read/ReadVariableOpslope_4/Read/ReadVariableOp slope_4/Adam/Read/ReadVariableOp"slope_4/Adam_1/Read/ReadVariableOpslope_5/Read/ReadVariableOp slope_5/Adam/Read/ReadVariableOp"slope_5/Adam_1/Read/ReadVariableOp*C
dtypes9
725
Щ
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: 
г
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
N*
T0*
_output_shapes
:*

axis 
Г
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
В
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
╞
save_2/RestoreV2/tensor_namesConst*
_output_shapes
:5*
dtype0*Ї
valueъBч5Bbeta1_powerBbeta2_powerBconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1Bconv2d_2/kernelBconv2d_2/kernel/AdamBconv2d_2/kernel/Adam_1Bconv2d_3/kernelBconv2d_3/kernel/AdamBconv2d_3/kernel/Adam_1Bconv2d_4/kernelBconv2d_4/kernel/AdamBconv2d_4/kernel/Adam_1B	interceptBintercept/AdamBintercept/Adam_1Bintercept_1Bintercept_1/AdamBintercept_1/Adam_1Bintercept_2Bintercept_2/AdamBintercept_2/Adam_1Bintercept_3Bintercept_3/AdamBintercept_3/Adam_1Bintercept_4Bintercept_4/AdamBintercept_4/Adam_1Bintercept_5Bintercept_5/AdamBintercept_5/Adam_1BslopeB
slope/AdamBslope/Adam_1Bslope_1Bslope_1/AdamBslope_1/Adam_1Bslope_2Bslope_2/AdamBslope_2/Adam_1Bslope_3Bslope_3/AdamBslope_3/Adam_1Bslope_4Bslope_4/AdamBslope_4/Adam_1Bslope_5Bslope_5/AdamBslope_5/Adam_1
╥
!save_2/RestoreV2/shape_and_slicesConst*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Я
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*ъ
_output_shapes╫
╘:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725
R
save_2/Identity_1Identitysave_2/RestoreV2*
T0*
_output_shapes
:
X
save_2/AssignVariableOpAssignVariableOpbeta1_powersave_2/Identity_1*
dtype0
T
save_2/Identity_2Identitysave_2/RestoreV2:1*
T0*
_output_shapes
:
Z
save_2/AssignVariableOp_1AssignVariableOpbeta2_powersave_2/Identity_2*
dtype0
T
save_2/Identity_3Identitysave_2/RestoreV2:2*
T0*
_output_shapes
:
\
save_2/AssignVariableOp_2AssignVariableOpconv2d/kernelsave_2/Identity_3*
dtype0
T
save_2/Identity_4Identitysave_2/RestoreV2:3*
T0*
_output_shapes
:
a
save_2/AssignVariableOp_3AssignVariableOpconv2d/kernel/Adamsave_2/Identity_4*
dtype0
T
save_2/Identity_5Identitysave_2/RestoreV2:4*
T0*
_output_shapes
:
c
save_2/AssignVariableOp_4AssignVariableOpconv2d/kernel/Adam_1save_2/Identity_5*
dtype0
T
save_2/Identity_6Identitysave_2/RestoreV2:5*
T0*
_output_shapes
:
^
save_2/AssignVariableOp_5AssignVariableOpconv2d_1/kernelsave_2/Identity_6*
dtype0
T
save_2/Identity_7Identitysave_2/RestoreV2:6*
T0*
_output_shapes
:
c
save_2/AssignVariableOp_6AssignVariableOpconv2d_1/kernel/Adamsave_2/Identity_7*
dtype0
T
save_2/Identity_8Identitysave_2/RestoreV2:7*
T0*
_output_shapes
:
e
save_2/AssignVariableOp_7AssignVariableOpconv2d_1/kernel/Adam_1save_2/Identity_8*
dtype0
T
save_2/Identity_9Identitysave_2/RestoreV2:8*
T0*
_output_shapes
:
^
save_2/AssignVariableOp_8AssignVariableOpconv2d_2/kernelsave_2/Identity_9*
dtype0
U
save_2/Identity_10Identitysave_2/RestoreV2:9*
T0*
_output_shapes
:
d
save_2/AssignVariableOp_9AssignVariableOpconv2d_2/kernel/Adamsave_2/Identity_10*
dtype0
V
save_2/Identity_11Identitysave_2/RestoreV2:10*
T0*
_output_shapes
:
g
save_2/AssignVariableOp_10AssignVariableOpconv2d_2/kernel/Adam_1save_2/Identity_11*
dtype0
V
save_2/Identity_12Identitysave_2/RestoreV2:11*
T0*
_output_shapes
:
`
save_2/AssignVariableOp_11AssignVariableOpconv2d_3/kernelsave_2/Identity_12*
dtype0
V
save_2/Identity_13Identitysave_2/RestoreV2:12*
T0*
_output_shapes
:
e
save_2/AssignVariableOp_12AssignVariableOpconv2d_3/kernel/Adamsave_2/Identity_13*
dtype0
V
save_2/Identity_14Identitysave_2/RestoreV2:13*
T0*
_output_shapes
:
g
save_2/AssignVariableOp_13AssignVariableOpconv2d_3/kernel/Adam_1save_2/Identity_14*
dtype0
V
save_2/Identity_15Identitysave_2/RestoreV2:14*
T0*
_output_shapes
:
`
save_2/AssignVariableOp_14AssignVariableOpconv2d_4/kernelsave_2/Identity_15*
dtype0
V
save_2/Identity_16Identitysave_2/RestoreV2:15*
T0*
_output_shapes
:
e
save_2/AssignVariableOp_15AssignVariableOpconv2d_4/kernel/Adamsave_2/Identity_16*
dtype0
V
save_2/Identity_17Identitysave_2/RestoreV2:16*
T0*
_output_shapes
:
g
save_2/AssignVariableOp_16AssignVariableOpconv2d_4/kernel/Adam_1save_2/Identity_17*
dtype0
V
save_2/Identity_18Identitysave_2/RestoreV2:17*
T0*
_output_shapes
:
Z
save_2/AssignVariableOp_17AssignVariableOp	interceptsave_2/Identity_18*
dtype0
V
save_2/Identity_19Identitysave_2/RestoreV2:18*
T0*
_output_shapes
:
_
save_2/AssignVariableOp_18AssignVariableOpintercept/Adamsave_2/Identity_19*
dtype0
V
save_2/Identity_20Identitysave_2/RestoreV2:19*
T0*
_output_shapes
:
a
save_2/AssignVariableOp_19AssignVariableOpintercept/Adam_1save_2/Identity_20*
dtype0
V
save_2/Identity_21Identitysave_2/RestoreV2:20*
T0*
_output_shapes
:
\
save_2/AssignVariableOp_20AssignVariableOpintercept_1save_2/Identity_21*
dtype0
V
save_2/Identity_22Identitysave_2/RestoreV2:21*
T0*
_output_shapes
:
a
save_2/AssignVariableOp_21AssignVariableOpintercept_1/Adamsave_2/Identity_22*
dtype0
V
save_2/Identity_23Identitysave_2/RestoreV2:22*
T0*
_output_shapes
:
c
save_2/AssignVariableOp_22AssignVariableOpintercept_1/Adam_1save_2/Identity_23*
dtype0
V
save_2/Identity_24Identitysave_2/RestoreV2:23*
T0*
_output_shapes
:
\
save_2/AssignVariableOp_23AssignVariableOpintercept_2save_2/Identity_24*
dtype0
V
save_2/Identity_25Identitysave_2/RestoreV2:24*
T0*
_output_shapes
:
a
save_2/AssignVariableOp_24AssignVariableOpintercept_2/Adamsave_2/Identity_25*
dtype0
V
save_2/Identity_26Identitysave_2/RestoreV2:25*
T0*
_output_shapes
:
c
save_2/AssignVariableOp_25AssignVariableOpintercept_2/Adam_1save_2/Identity_26*
dtype0
V
save_2/Identity_27Identitysave_2/RestoreV2:26*
T0*
_output_shapes
:
\
save_2/AssignVariableOp_26AssignVariableOpintercept_3save_2/Identity_27*
dtype0
V
save_2/Identity_28Identitysave_2/RestoreV2:27*
T0*
_output_shapes
:
a
save_2/AssignVariableOp_27AssignVariableOpintercept_3/Adamsave_2/Identity_28*
dtype0
V
save_2/Identity_29Identitysave_2/RestoreV2:28*
T0*
_output_shapes
:
c
save_2/AssignVariableOp_28AssignVariableOpintercept_3/Adam_1save_2/Identity_29*
dtype0
V
save_2/Identity_30Identitysave_2/RestoreV2:29*
T0*
_output_shapes
:
\
save_2/AssignVariableOp_29AssignVariableOpintercept_4save_2/Identity_30*
dtype0
V
save_2/Identity_31Identitysave_2/RestoreV2:30*
T0*
_output_shapes
:
a
save_2/AssignVariableOp_30AssignVariableOpintercept_4/Adamsave_2/Identity_31*
dtype0
V
save_2/Identity_32Identitysave_2/RestoreV2:31*
T0*
_output_shapes
:
c
save_2/AssignVariableOp_31AssignVariableOpintercept_4/Adam_1save_2/Identity_32*
dtype0
V
save_2/Identity_33Identitysave_2/RestoreV2:32*
T0*
_output_shapes
:
\
save_2/AssignVariableOp_32AssignVariableOpintercept_5save_2/Identity_33*
dtype0
V
save_2/Identity_34Identitysave_2/RestoreV2:33*
T0*
_output_shapes
:
a
save_2/AssignVariableOp_33AssignVariableOpintercept_5/Adamsave_2/Identity_34*
dtype0
V
save_2/Identity_35Identitysave_2/RestoreV2:34*
T0*
_output_shapes
:
c
save_2/AssignVariableOp_34AssignVariableOpintercept_5/Adam_1save_2/Identity_35*
dtype0
V
save_2/Identity_36Identitysave_2/RestoreV2:35*
T0*
_output_shapes
:
V
save_2/AssignVariableOp_35AssignVariableOpslopesave_2/Identity_36*
dtype0
V
save_2/Identity_37Identitysave_2/RestoreV2:36*
T0*
_output_shapes
:
[
save_2/AssignVariableOp_36AssignVariableOp
slope/Adamsave_2/Identity_37*
dtype0
V
save_2/Identity_38Identitysave_2/RestoreV2:37*
T0*
_output_shapes
:
]
save_2/AssignVariableOp_37AssignVariableOpslope/Adam_1save_2/Identity_38*
dtype0
V
save_2/Identity_39Identitysave_2/RestoreV2:38*
T0*
_output_shapes
:
X
save_2/AssignVariableOp_38AssignVariableOpslope_1save_2/Identity_39*
dtype0
V
save_2/Identity_40Identitysave_2/RestoreV2:39*
T0*
_output_shapes
:
]
save_2/AssignVariableOp_39AssignVariableOpslope_1/Adamsave_2/Identity_40*
dtype0
V
save_2/Identity_41Identitysave_2/RestoreV2:40*
T0*
_output_shapes
:
_
save_2/AssignVariableOp_40AssignVariableOpslope_1/Adam_1save_2/Identity_41*
dtype0
V
save_2/Identity_42Identitysave_2/RestoreV2:41*
T0*
_output_shapes
:
X
save_2/AssignVariableOp_41AssignVariableOpslope_2save_2/Identity_42*
dtype0
V
save_2/Identity_43Identitysave_2/RestoreV2:42*
T0*
_output_shapes
:
]
save_2/AssignVariableOp_42AssignVariableOpslope_2/Adamsave_2/Identity_43*
dtype0
V
save_2/Identity_44Identitysave_2/RestoreV2:43*
T0*
_output_shapes
:
_
save_2/AssignVariableOp_43AssignVariableOpslope_2/Adam_1save_2/Identity_44*
dtype0
V
save_2/Identity_45Identitysave_2/RestoreV2:44*
T0*
_output_shapes
:
X
save_2/AssignVariableOp_44AssignVariableOpslope_3save_2/Identity_45*
dtype0
V
save_2/Identity_46Identitysave_2/RestoreV2:45*
T0*
_output_shapes
:
]
save_2/AssignVariableOp_45AssignVariableOpslope_3/Adamsave_2/Identity_46*
dtype0
V
save_2/Identity_47Identitysave_2/RestoreV2:46*
T0*
_output_shapes
:
_
save_2/AssignVariableOp_46AssignVariableOpslope_3/Adam_1save_2/Identity_47*
dtype0
V
save_2/Identity_48Identitysave_2/RestoreV2:47*
T0*
_output_shapes
:
X
save_2/AssignVariableOp_47AssignVariableOpslope_4save_2/Identity_48*
dtype0
V
save_2/Identity_49Identitysave_2/RestoreV2:48*
T0*
_output_shapes
:
]
save_2/AssignVariableOp_48AssignVariableOpslope_4/Adamsave_2/Identity_49*
dtype0
V
save_2/Identity_50Identitysave_2/RestoreV2:49*
T0*
_output_shapes
:
_
save_2/AssignVariableOp_49AssignVariableOpslope_4/Adam_1save_2/Identity_50*
dtype0
V
save_2/Identity_51Identitysave_2/RestoreV2:50*
T0*
_output_shapes
:
X
save_2/AssignVariableOp_50AssignVariableOpslope_5save_2/Identity_51*
dtype0
V
save_2/Identity_52Identitysave_2/RestoreV2:51*
T0*
_output_shapes
:
]
save_2/AssignVariableOp_51AssignVariableOpslope_5/Adamsave_2/Identity_52*
dtype0
V
save_2/Identity_53Identitysave_2/RestoreV2:52*
T0*
_output_shapes
:
_
save_2/AssignVariableOp_52AssignVariableOpslope_5/Adam_1save_2/Identity_53*
dtype0
С
save_2/restore_shardNoOp^save_2/AssignVariableOp^save_2/AssignVariableOp_1^save_2/AssignVariableOp_10^save_2/AssignVariableOp_11^save_2/AssignVariableOp_12^save_2/AssignVariableOp_13^save_2/AssignVariableOp_14^save_2/AssignVariableOp_15^save_2/AssignVariableOp_16^save_2/AssignVariableOp_17^save_2/AssignVariableOp_18^save_2/AssignVariableOp_19^save_2/AssignVariableOp_2^save_2/AssignVariableOp_20^save_2/AssignVariableOp_21^save_2/AssignVariableOp_22^save_2/AssignVariableOp_23^save_2/AssignVariableOp_24^save_2/AssignVariableOp_25^save_2/AssignVariableOp_26^save_2/AssignVariableOp_27^save_2/AssignVariableOp_28^save_2/AssignVariableOp_29^save_2/AssignVariableOp_3^save_2/AssignVariableOp_30^save_2/AssignVariableOp_31^save_2/AssignVariableOp_32^save_2/AssignVariableOp_33^save_2/AssignVariableOp_34^save_2/AssignVariableOp_35^save_2/AssignVariableOp_36^save_2/AssignVariableOp_37^save_2/AssignVariableOp_38^save_2/AssignVariableOp_39^save_2/AssignVariableOp_4^save_2/AssignVariableOp_40^save_2/AssignVariableOp_41^save_2/AssignVariableOp_42^save_2/AssignVariableOp_43^save_2/AssignVariableOp_44^save_2/AssignVariableOp_45^save_2/AssignVariableOp_46^save_2/AssignVariableOp_47^save_2/AssignVariableOp_48^save_2/AssignVariableOp_49^save_2/AssignVariableOp_5^save_2/AssignVariableOp_50^save_2/AssignVariableOp_51^save_2/AssignVariableOp_52^save_2/AssignVariableOp_6^save_2/AssignVariableOp_7^save_2/AssignVariableOp_8^save_2/AssignVariableOp_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
_output_shapes
: *
dtype0*
shape: 
Ж
save_3/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d3f0a53d135944ccbb74f4f4d7ce8ab3/part
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_3/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_3/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
Е
save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
├
save_3/SaveV2/tensor_namesConst*
_output_shapes
:5*
dtype0*Ї
valueъBч5Bbeta1_powerBbeta2_powerBconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1Bconv2d_2/kernelBconv2d_2/kernel/AdamBconv2d_2/kernel/Adam_1Bconv2d_3/kernelBconv2d_3/kernel/AdamBconv2d_3/kernel/Adam_1Bconv2d_4/kernelBconv2d_4/kernel/AdamBconv2d_4/kernel/Adam_1B	interceptBintercept/AdamBintercept/Adam_1Bintercept_1Bintercept_1/AdamBintercept_1/Adam_1Bintercept_2Bintercept_2/AdamBintercept_2/Adam_1Bintercept_3Bintercept_3/AdamBintercept_3/Adam_1Bintercept_4Bintercept_4/AdamBintercept_4/Adam_1Bintercept_5Bintercept_5/AdamBintercept_5/Adam_1BslopeB
slope/AdamBslope/Adam_1Bslope_1Bslope_1/AdamBslope_1/Adam_1Bslope_2Bslope_2/AdamBslope_2/Adam_1Bslope_3Bslope_3/AdamBslope_3/Adam_1Bslope_4Bslope_4/AdamBslope_4/Adam_1Bslope_5Bslope_5/AdamBslope_5/Adam_1
╧
save_3/SaveV2/shape_and_slicesConst*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
│
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_power/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOp&conv2d/kernel/Adam/Read/ReadVariableOp(conv2d/kernel/Adam_1/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp(conv2d_1/kernel/Adam/Read/ReadVariableOp*conv2d_1/kernel/Adam_1/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp(conv2d_2/kernel/Adam/Read/ReadVariableOp*conv2d_2/kernel/Adam_1/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp(conv2d_3/kernel/Adam/Read/ReadVariableOp*conv2d_3/kernel/Adam_1/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp(conv2d_4/kernel/Adam/Read/ReadVariableOp*conv2d_4/kernel/Adam_1/Read/ReadVariableOpintercept/Read/ReadVariableOp"intercept/Adam/Read/ReadVariableOp$intercept/Adam_1/Read/ReadVariableOpintercept_1/Read/ReadVariableOp$intercept_1/Adam/Read/ReadVariableOp&intercept_1/Adam_1/Read/ReadVariableOpintercept_2/Read/ReadVariableOp$intercept_2/Adam/Read/ReadVariableOp&intercept_2/Adam_1/Read/ReadVariableOpintercept_3/Read/ReadVariableOp$intercept_3/Adam/Read/ReadVariableOp&intercept_3/Adam_1/Read/ReadVariableOpintercept_4/Read/ReadVariableOp$intercept_4/Adam/Read/ReadVariableOp&intercept_4/Adam_1/Read/ReadVariableOpintercept_5/Read/ReadVariableOp$intercept_5/Adam/Read/ReadVariableOp&intercept_5/Adam_1/Read/ReadVariableOpslope/Read/ReadVariableOpslope/Adam/Read/ReadVariableOp slope/Adam_1/Read/ReadVariableOpslope_1/Read/ReadVariableOp slope_1/Adam/Read/ReadVariableOp"slope_1/Adam_1/Read/ReadVariableOpslope_2/Read/ReadVariableOp slope_2/Adam/Read/ReadVariableOp"slope_2/Adam_1/Read/ReadVariableOpslope_3/Read/ReadVariableOp slope_3/Adam/Read/ReadVariableOp"slope_3/Adam_1/Read/ReadVariableOpslope_4/Read/ReadVariableOp slope_4/Adam/Read/ReadVariableOp"slope_4/Adam_1/Read/ReadVariableOpslope_5/Read/ReadVariableOp slope_5/Adam/Read/ReadVariableOp"slope_5/Adam_1/Read/ReadVariableOp*C
dtypes9
725
Щ
save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: 
г
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
N*
T0*
_output_shapes
:*

axis 
Г
save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(
В
save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
T0*
_output_shapes
: 
╞
save_3/RestoreV2/tensor_namesConst*
_output_shapes
:5*
dtype0*Ї
valueъBч5Bbeta1_powerBbeta2_powerBconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1Bconv2d_2/kernelBconv2d_2/kernel/AdamBconv2d_2/kernel/Adam_1Bconv2d_3/kernelBconv2d_3/kernel/AdamBconv2d_3/kernel/Adam_1Bconv2d_4/kernelBconv2d_4/kernel/AdamBconv2d_4/kernel/Adam_1B	interceptBintercept/AdamBintercept/Adam_1Bintercept_1Bintercept_1/AdamBintercept_1/Adam_1Bintercept_2Bintercept_2/AdamBintercept_2/Adam_1Bintercept_3Bintercept_3/AdamBintercept_3/Adam_1Bintercept_4Bintercept_4/AdamBintercept_4/Adam_1Bintercept_5Bintercept_5/AdamBintercept_5/Adam_1BslopeB
slope/AdamBslope/Adam_1Bslope_1Bslope_1/AdamBslope_1/Adam_1Bslope_2Bslope_2/AdamBslope_2/Adam_1Bslope_3Bslope_3/AdamBslope_3/Adam_1Bslope_4Bslope_4/AdamBslope_4/Adam_1Bslope_5Bslope_5/AdamBslope_5/Adam_1
╥
!save_3/RestoreV2/shape_and_slicesConst*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Я
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*ъ
_output_shapes╫
╘:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725
R
save_3/Identity_1Identitysave_3/RestoreV2*
T0*
_output_shapes
:
X
save_3/AssignVariableOpAssignVariableOpbeta1_powersave_3/Identity_1*
dtype0
T
save_3/Identity_2Identitysave_3/RestoreV2:1*
T0*
_output_shapes
:
Z
save_3/AssignVariableOp_1AssignVariableOpbeta2_powersave_3/Identity_2*
dtype0
T
save_3/Identity_3Identitysave_3/RestoreV2:2*
T0*
_output_shapes
:
\
save_3/AssignVariableOp_2AssignVariableOpconv2d/kernelsave_3/Identity_3*
dtype0
T
save_3/Identity_4Identitysave_3/RestoreV2:3*
T0*
_output_shapes
:
a
save_3/AssignVariableOp_3AssignVariableOpconv2d/kernel/Adamsave_3/Identity_4*
dtype0
T
save_3/Identity_5Identitysave_3/RestoreV2:4*
T0*
_output_shapes
:
c
save_3/AssignVariableOp_4AssignVariableOpconv2d/kernel/Adam_1save_3/Identity_5*
dtype0
T
save_3/Identity_6Identitysave_3/RestoreV2:5*
T0*
_output_shapes
:
^
save_3/AssignVariableOp_5AssignVariableOpconv2d_1/kernelsave_3/Identity_6*
dtype0
T
save_3/Identity_7Identitysave_3/RestoreV2:6*
T0*
_output_shapes
:
c
save_3/AssignVariableOp_6AssignVariableOpconv2d_1/kernel/Adamsave_3/Identity_7*
dtype0
T
save_3/Identity_8Identitysave_3/RestoreV2:7*
T0*
_output_shapes
:
e
save_3/AssignVariableOp_7AssignVariableOpconv2d_1/kernel/Adam_1save_3/Identity_8*
dtype0
T
save_3/Identity_9Identitysave_3/RestoreV2:8*
T0*
_output_shapes
:
^
save_3/AssignVariableOp_8AssignVariableOpconv2d_2/kernelsave_3/Identity_9*
dtype0
U
save_3/Identity_10Identitysave_3/RestoreV2:9*
T0*
_output_shapes
:
d
save_3/AssignVariableOp_9AssignVariableOpconv2d_2/kernel/Adamsave_3/Identity_10*
dtype0
V
save_3/Identity_11Identitysave_3/RestoreV2:10*
T0*
_output_shapes
:
g
save_3/AssignVariableOp_10AssignVariableOpconv2d_2/kernel/Adam_1save_3/Identity_11*
dtype0
V
save_3/Identity_12Identitysave_3/RestoreV2:11*
T0*
_output_shapes
:
`
save_3/AssignVariableOp_11AssignVariableOpconv2d_3/kernelsave_3/Identity_12*
dtype0
V
save_3/Identity_13Identitysave_3/RestoreV2:12*
T0*
_output_shapes
:
e
save_3/AssignVariableOp_12AssignVariableOpconv2d_3/kernel/Adamsave_3/Identity_13*
dtype0
V
save_3/Identity_14Identitysave_3/RestoreV2:13*
T0*
_output_shapes
:
g
save_3/AssignVariableOp_13AssignVariableOpconv2d_3/kernel/Adam_1save_3/Identity_14*
dtype0
V
save_3/Identity_15Identitysave_3/RestoreV2:14*
T0*
_output_shapes
:
`
save_3/AssignVariableOp_14AssignVariableOpconv2d_4/kernelsave_3/Identity_15*
dtype0
V
save_3/Identity_16Identitysave_3/RestoreV2:15*
T0*
_output_shapes
:
e
save_3/AssignVariableOp_15AssignVariableOpconv2d_4/kernel/Adamsave_3/Identity_16*
dtype0
V
save_3/Identity_17Identitysave_3/RestoreV2:16*
T0*
_output_shapes
:
g
save_3/AssignVariableOp_16AssignVariableOpconv2d_4/kernel/Adam_1save_3/Identity_17*
dtype0
V
save_3/Identity_18Identitysave_3/RestoreV2:17*
T0*
_output_shapes
:
Z
save_3/AssignVariableOp_17AssignVariableOp	interceptsave_3/Identity_18*
dtype0
V
save_3/Identity_19Identitysave_3/RestoreV2:18*
T0*
_output_shapes
:
_
save_3/AssignVariableOp_18AssignVariableOpintercept/Adamsave_3/Identity_19*
dtype0
V
save_3/Identity_20Identitysave_3/RestoreV2:19*
T0*
_output_shapes
:
a
save_3/AssignVariableOp_19AssignVariableOpintercept/Adam_1save_3/Identity_20*
dtype0
V
save_3/Identity_21Identitysave_3/RestoreV2:20*
T0*
_output_shapes
:
\
save_3/AssignVariableOp_20AssignVariableOpintercept_1save_3/Identity_21*
dtype0
V
save_3/Identity_22Identitysave_3/RestoreV2:21*
T0*
_output_shapes
:
a
save_3/AssignVariableOp_21AssignVariableOpintercept_1/Adamsave_3/Identity_22*
dtype0
V
save_3/Identity_23Identitysave_3/RestoreV2:22*
T0*
_output_shapes
:
c
save_3/AssignVariableOp_22AssignVariableOpintercept_1/Adam_1save_3/Identity_23*
dtype0
V
save_3/Identity_24Identitysave_3/RestoreV2:23*
T0*
_output_shapes
:
\
save_3/AssignVariableOp_23AssignVariableOpintercept_2save_3/Identity_24*
dtype0
V
save_3/Identity_25Identitysave_3/RestoreV2:24*
T0*
_output_shapes
:
a
save_3/AssignVariableOp_24AssignVariableOpintercept_2/Adamsave_3/Identity_25*
dtype0
V
save_3/Identity_26Identitysave_3/RestoreV2:25*
T0*
_output_shapes
:
c
save_3/AssignVariableOp_25AssignVariableOpintercept_2/Adam_1save_3/Identity_26*
dtype0
V
save_3/Identity_27Identitysave_3/RestoreV2:26*
T0*
_output_shapes
:
\
save_3/AssignVariableOp_26AssignVariableOpintercept_3save_3/Identity_27*
dtype0
V
save_3/Identity_28Identitysave_3/RestoreV2:27*
T0*
_output_shapes
:
a
save_3/AssignVariableOp_27AssignVariableOpintercept_3/Adamsave_3/Identity_28*
dtype0
V
save_3/Identity_29Identitysave_3/RestoreV2:28*
T0*
_output_shapes
:
c
save_3/AssignVariableOp_28AssignVariableOpintercept_3/Adam_1save_3/Identity_29*
dtype0
V
save_3/Identity_30Identitysave_3/RestoreV2:29*
T0*
_output_shapes
:
\
save_3/AssignVariableOp_29AssignVariableOpintercept_4save_3/Identity_30*
dtype0
V
save_3/Identity_31Identitysave_3/RestoreV2:30*
T0*
_output_shapes
:
a
save_3/AssignVariableOp_30AssignVariableOpintercept_4/Adamsave_3/Identity_31*
dtype0
V
save_3/Identity_32Identitysave_3/RestoreV2:31*
T0*
_output_shapes
:
c
save_3/AssignVariableOp_31AssignVariableOpintercept_4/Adam_1save_3/Identity_32*
dtype0
V
save_3/Identity_33Identitysave_3/RestoreV2:32*
T0*
_output_shapes
:
\
save_3/AssignVariableOp_32AssignVariableOpintercept_5save_3/Identity_33*
dtype0
V
save_3/Identity_34Identitysave_3/RestoreV2:33*
T0*
_output_shapes
:
a
save_3/AssignVariableOp_33AssignVariableOpintercept_5/Adamsave_3/Identity_34*
dtype0
V
save_3/Identity_35Identitysave_3/RestoreV2:34*
T0*
_output_shapes
:
c
save_3/AssignVariableOp_34AssignVariableOpintercept_5/Adam_1save_3/Identity_35*
dtype0
V
save_3/Identity_36Identitysave_3/RestoreV2:35*
T0*
_output_shapes
:
V
save_3/AssignVariableOp_35AssignVariableOpslopesave_3/Identity_36*
dtype0
V
save_3/Identity_37Identitysave_3/RestoreV2:36*
T0*
_output_shapes
:
[
save_3/AssignVariableOp_36AssignVariableOp
slope/Adamsave_3/Identity_37*
dtype0
V
save_3/Identity_38Identitysave_3/RestoreV2:37*
T0*
_output_shapes
:
]
save_3/AssignVariableOp_37AssignVariableOpslope/Adam_1save_3/Identity_38*
dtype0
V
save_3/Identity_39Identitysave_3/RestoreV2:38*
T0*
_output_shapes
:
X
save_3/AssignVariableOp_38AssignVariableOpslope_1save_3/Identity_39*
dtype0
V
save_3/Identity_40Identitysave_3/RestoreV2:39*
T0*
_output_shapes
:
]
save_3/AssignVariableOp_39AssignVariableOpslope_1/Adamsave_3/Identity_40*
dtype0
V
save_3/Identity_41Identitysave_3/RestoreV2:40*
T0*
_output_shapes
:
_
save_3/AssignVariableOp_40AssignVariableOpslope_1/Adam_1save_3/Identity_41*
dtype0
V
save_3/Identity_42Identitysave_3/RestoreV2:41*
T0*
_output_shapes
:
X
save_3/AssignVariableOp_41AssignVariableOpslope_2save_3/Identity_42*
dtype0
V
save_3/Identity_43Identitysave_3/RestoreV2:42*
T0*
_output_shapes
:
]
save_3/AssignVariableOp_42AssignVariableOpslope_2/Adamsave_3/Identity_43*
dtype0
V
save_3/Identity_44Identitysave_3/RestoreV2:43*
T0*
_output_shapes
:
_
save_3/AssignVariableOp_43AssignVariableOpslope_2/Adam_1save_3/Identity_44*
dtype0
V
save_3/Identity_45Identitysave_3/RestoreV2:44*
T0*
_output_shapes
:
X
save_3/AssignVariableOp_44AssignVariableOpslope_3save_3/Identity_45*
dtype0
V
save_3/Identity_46Identitysave_3/RestoreV2:45*
T0*
_output_shapes
:
]
save_3/AssignVariableOp_45AssignVariableOpslope_3/Adamsave_3/Identity_46*
dtype0
V
save_3/Identity_47Identitysave_3/RestoreV2:46*
T0*
_output_shapes
:
_
save_3/AssignVariableOp_46AssignVariableOpslope_3/Adam_1save_3/Identity_47*
dtype0
V
save_3/Identity_48Identitysave_3/RestoreV2:47*
T0*
_output_shapes
:
X
save_3/AssignVariableOp_47AssignVariableOpslope_4save_3/Identity_48*
dtype0
V
save_3/Identity_49Identitysave_3/RestoreV2:48*
T0*
_output_shapes
:
]
save_3/AssignVariableOp_48AssignVariableOpslope_4/Adamsave_3/Identity_49*
dtype0
V
save_3/Identity_50Identitysave_3/RestoreV2:49*
T0*
_output_shapes
:
_
save_3/AssignVariableOp_49AssignVariableOpslope_4/Adam_1save_3/Identity_50*
dtype0
V
save_3/Identity_51Identitysave_3/RestoreV2:50*
T0*
_output_shapes
:
X
save_3/AssignVariableOp_50AssignVariableOpslope_5save_3/Identity_51*
dtype0
V
save_3/Identity_52Identitysave_3/RestoreV2:51*
T0*
_output_shapes
:
]
save_3/AssignVariableOp_51AssignVariableOpslope_5/Adamsave_3/Identity_52*
dtype0
V
save_3/Identity_53Identitysave_3/RestoreV2:52*
T0*
_output_shapes
:
_
save_3/AssignVariableOp_52AssignVariableOpslope_5/Adam_1save_3/Identity_53*
dtype0
С
save_3/restore_shardNoOp^save_3/AssignVariableOp^save_3/AssignVariableOp_1^save_3/AssignVariableOp_10^save_3/AssignVariableOp_11^save_3/AssignVariableOp_12^save_3/AssignVariableOp_13^save_3/AssignVariableOp_14^save_3/AssignVariableOp_15^save_3/AssignVariableOp_16^save_3/AssignVariableOp_17^save_3/AssignVariableOp_18^save_3/AssignVariableOp_19^save_3/AssignVariableOp_2^save_3/AssignVariableOp_20^save_3/AssignVariableOp_21^save_3/AssignVariableOp_22^save_3/AssignVariableOp_23^save_3/AssignVariableOp_24^save_3/AssignVariableOp_25^save_3/AssignVariableOp_26^save_3/AssignVariableOp_27^save_3/AssignVariableOp_28^save_3/AssignVariableOp_29^save_3/AssignVariableOp_3^save_3/AssignVariableOp_30^save_3/AssignVariableOp_31^save_3/AssignVariableOp_32^save_3/AssignVariableOp_33^save_3/AssignVariableOp_34^save_3/AssignVariableOp_35^save_3/AssignVariableOp_36^save_3/AssignVariableOp_37^save_3/AssignVariableOp_38^save_3/AssignVariableOp_39^save_3/AssignVariableOp_4^save_3/AssignVariableOp_40^save_3/AssignVariableOp_41^save_3/AssignVariableOp_42^save_3/AssignVariableOp_43^save_3/AssignVariableOp_44^save_3/AssignVariableOp_45^save_3/AssignVariableOp_46^save_3/AssignVariableOp_47^save_3/AssignVariableOp_48^save_3/AssignVariableOp_49^save_3/AssignVariableOp_5^save_3/AssignVariableOp_50^save_3/AssignVariableOp_51^save_3/AssignVariableOp_52^save_3/AssignVariableOp_6^save_3/AssignVariableOp_7^save_3/AssignVariableOp_8^save_3/AssignVariableOp_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
_output_shapes
: *
dtype0*
shape: 
Ж
save_4/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ed1c85f2bdc24c14a01f4af0d953ef86/part
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_4/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_4/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
Е
save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
├
save_4/SaveV2/tensor_namesConst*
_output_shapes
:5*
dtype0*Ї
valueъBч5Bbeta1_powerBbeta2_powerBconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1Bconv2d_2/kernelBconv2d_2/kernel/AdamBconv2d_2/kernel/Adam_1Bconv2d_3/kernelBconv2d_3/kernel/AdamBconv2d_3/kernel/Adam_1Bconv2d_4/kernelBconv2d_4/kernel/AdamBconv2d_4/kernel/Adam_1B	interceptBintercept/AdamBintercept/Adam_1Bintercept_1Bintercept_1/AdamBintercept_1/Adam_1Bintercept_2Bintercept_2/AdamBintercept_2/Adam_1Bintercept_3Bintercept_3/AdamBintercept_3/Adam_1Bintercept_4Bintercept_4/AdamBintercept_4/Adam_1Bintercept_5Bintercept_5/AdamBintercept_5/Adam_1BslopeB
slope/AdamBslope/Adam_1Bslope_1Bslope_1/AdamBslope_1/Adam_1Bslope_2Bslope_2/AdamBslope_2/Adam_1Bslope_3Bslope_3/AdamBslope_3/Adam_1Bslope_4Bslope_4/AdamBslope_4/Adam_1Bslope_5Bslope_5/AdamBslope_5/Adam_1
╧
save_4/SaveV2/shape_and_slicesConst*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
│
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_power/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOp&conv2d/kernel/Adam/Read/ReadVariableOp(conv2d/kernel/Adam_1/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp(conv2d_1/kernel/Adam/Read/ReadVariableOp*conv2d_1/kernel/Adam_1/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp(conv2d_2/kernel/Adam/Read/ReadVariableOp*conv2d_2/kernel/Adam_1/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp(conv2d_3/kernel/Adam/Read/ReadVariableOp*conv2d_3/kernel/Adam_1/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp(conv2d_4/kernel/Adam/Read/ReadVariableOp*conv2d_4/kernel/Adam_1/Read/ReadVariableOpintercept/Read/ReadVariableOp"intercept/Adam/Read/ReadVariableOp$intercept/Adam_1/Read/ReadVariableOpintercept_1/Read/ReadVariableOp$intercept_1/Adam/Read/ReadVariableOp&intercept_1/Adam_1/Read/ReadVariableOpintercept_2/Read/ReadVariableOp$intercept_2/Adam/Read/ReadVariableOp&intercept_2/Adam_1/Read/ReadVariableOpintercept_3/Read/ReadVariableOp$intercept_3/Adam/Read/ReadVariableOp&intercept_3/Adam_1/Read/ReadVariableOpintercept_4/Read/ReadVariableOp$intercept_4/Adam/Read/ReadVariableOp&intercept_4/Adam_1/Read/ReadVariableOpintercept_5/Read/ReadVariableOp$intercept_5/Adam/Read/ReadVariableOp&intercept_5/Adam_1/Read/ReadVariableOpslope/Read/ReadVariableOpslope/Adam/Read/ReadVariableOp slope/Adam_1/Read/ReadVariableOpslope_1/Read/ReadVariableOp slope_1/Adam/Read/ReadVariableOp"slope_1/Adam_1/Read/ReadVariableOpslope_2/Read/ReadVariableOp slope_2/Adam/Read/ReadVariableOp"slope_2/Adam_1/Read/ReadVariableOpslope_3/Read/ReadVariableOp slope_3/Adam/Read/ReadVariableOp"slope_3/Adam_1/Read/ReadVariableOpslope_4/Read/ReadVariableOp slope_4/Adam/Read/ReadVariableOp"slope_4/Adam_1/Read/ReadVariableOpslope_5/Read/ReadVariableOp slope_5/Adam/Read/ReadVariableOp"slope_5/Adam_1/Read/ReadVariableOp*C
dtypes9
725
Щ
save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*)
_class
loc:@save_4/ShardedFilename*
_output_shapes
: 
г
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*
N*
T0*
_output_shapes
:*

axis 
Г
save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(
В
save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
T0*
_output_shapes
: 
╞
save_4/RestoreV2/tensor_namesConst*
_output_shapes
:5*
dtype0*Ї
valueъBч5Bbeta1_powerBbeta2_powerBconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1Bconv2d_2/kernelBconv2d_2/kernel/AdamBconv2d_2/kernel/Adam_1Bconv2d_3/kernelBconv2d_3/kernel/AdamBconv2d_3/kernel/Adam_1Bconv2d_4/kernelBconv2d_4/kernel/AdamBconv2d_4/kernel/Adam_1B	interceptBintercept/AdamBintercept/Adam_1Bintercept_1Bintercept_1/AdamBintercept_1/Adam_1Bintercept_2Bintercept_2/AdamBintercept_2/Adam_1Bintercept_3Bintercept_3/AdamBintercept_3/Adam_1Bintercept_4Bintercept_4/AdamBintercept_4/Adam_1Bintercept_5Bintercept_5/AdamBintercept_5/Adam_1BslopeB
slope/AdamBslope/Adam_1Bslope_1Bslope_1/AdamBslope_1/Adam_1Bslope_2Bslope_2/AdamBslope_2/Adam_1Bslope_3Bslope_3/AdamBslope_3/Adam_1Bslope_4Bslope_4/AdamBslope_4/Adam_1Bslope_5Bslope_5/AdamBslope_5/Adam_1
╥
!save_4/RestoreV2/shape_and_slicesConst*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Я
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*ъ
_output_shapes╫
╘:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725
R
save_4/Identity_1Identitysave_4/RestoreV2*
T0*
_output_shapes
:
X
save_4/AssignVariableOpAssignVariableOpbeta1_powersave_4/Identity_1*
dtype0
T
save_4/Identity_2Identitysave_4/RestoreV2:1*
T0*
_output_shapes
:
Z
save_4/AssignVariableOp_1AssignVariableOpbeta2_powersave_4/Identity_2*
dtype0
T
save_4/Identity_3Identitysave_4/RestoreV2:2*
T0*
_output_shapes
:
\
save_4/AssignVariableOp_2AssignVariableOpconv2d/kernelsave_4/Identity_3*
dtype0
T
save_4/Identity_4Identitysave_4/RestoreV2:3*
T0*
_output_shapes
:
a
save_4/AssignVariableOp_3AssignVariableOpconv2d/kernel/Adamsave_4/Identity_4*
dtype0
T
save_4/Identity_5Identitysave_4/RestoreV2:4*
T0*
_output_shapes
:
c
save_4/AssignVariableOp_4AssignVariableOpconv2d/kernel/Adam_1save_4/Identity_5*
dtype0
T
save_4/Identity_6Identitysave_4/RestoreV2:5*
T0*
_output_shapes
:
^
save_4/AssignVariableOp_5AssignVariableOpconv2d_1/kernelsave_4/Identity_6*
dtype0
T
save_4/Identity_7Identitysave_4/RestoreV2:6*
T0*
_output_shapes
:
c
save_4/AssignVariableOp_6AssignVariableOpconv2d_1/kernel/Adamsave_4/Identity_7*
dtype0
T
save_4/Identity_8Identitysave_4/RestoreV2:7*
T0*
_output_shapes
:
e
save_4/AssignVariableOp_7AssignVariableOpconv2d_1/kernel/Adam_1save_4/Identity_8*
dtype0
T
save_4/Identity_9Identitysave_4/RestoreV2:8*
T0*
_output_shapes
:
^
save_4/AssignVariableOp_8AssignVariableOpconv2d_2/kernelsave_4/Identity_9*
dtype0
U
save_4/Identity_10Identitysave_4/RestoreV2:9*
T0*
_output_shapes
:
d
save_4/AssignVariableOp_9AssignVariableOpconv2d_2/kernel/Adamsave_4/Identity_10*
dtype0
V
save_4/Identity_11Identitysave_4/RestoreV2:10*
T0*
_output_shapes
:
g
save_4/AssignVariableOp_10AssignVariableOpconv2d_2/kernel/Adam_1save_4/Identity_11*
dtype0
V
save_4/Identity_12Identitysave_4/RestoreV2:11*
T0*
_output_shapes
:
`
save_4/AssignVariableOp_11AssignVariableOpconv2d_3/kernelsave_4/Identity_12*
dtype0
V
save_4/Identity_13Identitysave_4/RestoreV2:12*
T0*
_output_shapes
:
e
save_4/AssignVariableOp_12AssignVariableOpconv2d_3/kernel/Adamsave_4/Identity_13*
dtype0
V
save_4/Identity_14Identitysave_4/RestoreV2:13*
T0*
_output_shapes
:
g
save_4/AssignVariableOp_13AssignVariableOpconv2d_3/kernel/Adam_1save_4/Identity_14*
dtype0
V
save_4/Identity_15Identitysave_4/RestoreV2:14*
T0*
_output_shapes
:
`
save_4/AssignVariableOp_14AssignVariableOpconv2d_4/kernelsave_4/Identity_15*
dtype0
V
save_4/Identity_16Identitysave_4/RestoreV2:15*
T0*
_output_shapes
:
e
save_4/AssignVariableOp_15AssignVariableOpconv2d_4/kernel/Adamsave_4/Identity_16*
dtype0
V
save_4/Identity_17Identitysave_4/RestoreV2:16*
T0*
_output_shapes
:
g
save_4/AssignVariableOp_16AssignVariableOpconv2d_4/kernel/Adam_1save_4/Identity_17*
dtype0
V
save_4/Identity_18Identitysave_4/RestoreV2:17*
T0*
_output_shapes
:
Z
save_4/AssignVariableOp_17AssignVariableOp	interceptsave_4/Identity_18*
dtype0
V
save_4/Identity_19Identitysave_4/RestoreV2:18*
T0*
_output_shapes
:
_
save_4/AssignVariableOp_18AssignVariableOpintercept/Adamsave_4/Identity_19*
dtype0
V
save_4/Identity_20Identitysave_4/RestoreV2:19*
T0*
_output_shapes
:
a
save_4/AssignVariableOp_19AssignVariableOpintercept/Adam_1save_4/Identity_20*
dtype0
V
save_4/Identity_21Identitysave_4/RestoreV2:20*
T0*
_output_shapes
:
\
save_4/AssignVariableOp_20AssignVariableOpintercept_1save_4/Identity_21*
dtype0
V
save_4/Identity_22Identitysave_4/RestoreV2:21*
T0*
_output_shapes
:
a
save_4/AssignVariableOp_21AssignVariableOpintercept_1/Adamsave_4/Identity_22*
dtype0
V
save_4/Identity_23Identitysave_4/RestoreV2:22*
T0*
_output_shapes
:
c
save_4/AssignVariableOp_22AssignVariableOpintercept_1/Adam_1save_4/Identity_23*
dtype0
V
save_4/Identity_24Identitysave_4/RestoreV2:23*
T0*
_output_shapes
:
\
save_4/AssignVariableOp_23AssignVariableOpintercept_2save_4/Identity_24*
dtype0
V
save_4/Identity_25Identitysave_4/RestoreV2:24*
T0*
_output_shapes
:
a
save_4/AssignVariableOp_24AssignVariableOpintercept_2/Adamsave_4/Identity_25*
dtype0
V
save_4/Identity_26Identitysave_4/RestoreV2:25*
T0*
_output_shapes
:
c
save_4/AssignVariableOp_25AssignVariableOpintercept_2/Adam_1save_4/Identity_26*
dtype0
V
save_4/Identity_27Identitysave_4/RestoreV2:26*
T0*
_output_shapes
:
\
save_4/AssignVariableOp_26AssignVariableOpintercept_3save_4/Identity_27*
dtype0
V
save_4/Identity_28Identitysave_4/RestoreV2:27*
T0*
_output_shapes
:
a
save_4/AssignVariableOp_27AssignVariableOpintercept_3/Adamsave_4/Identity_28*
dtype0
V
save_4/Identity_29Identitysave_4/RestoreV2:28*
T0*
_output_shapes
:
c
save_4/AssignVariableOp_28AssignVariableOpintercept_3/Adam_1save_4/Identity_29*
dtype0
V
save_4/Identity_30Identitysave_4/RestoreV2:29*
T0*
_output_shapes
:
\
save_4/AssignVariableOp_29AssignVariableOpintercept_4save_4/Identity_30*
dtype0
V
save_4/Identity_31Identitysave_4/RestoreV2:30*
T0*
_output_shapes
:
a
save_4/AssignVariableOp_30AssignVariableOpintercept_4/Adamsave_4/Identity_31*
dtype0
V
save_4/Identity_32Identitysave_4/RestoreV2:31*
T0*
_output_shapes
:
c
save_4/AssignVariableOp_31AssignVariableOpintercept_4/Adam_1save_4/Identity_32*
dtype0
V
save_4/Identity_33Identitysave_4/RestoreV2:32*
T0*
_output_shapes
:
\
save_4/AssignVariableOp_32AssignVariableOpintercept_5save_4/Identity_33*
dtype0
V
save_4/Identity_34Identitysave_4/RestoreV2:33*
T0*
_output_shapes
:
a
save_4/AssignVariableOp_33AssignVariableOpintercept_5/Adamsave_4/Identity_34*
dtype0
V
save_4/Identity_35Identitysave_4/RestoreV2:34*
T0*
_output_shapes
:
c
save_4/AssignVariableOp_34AssignVariableOpintercept_5/Adam_1save_4/Identity_35*
dtype0
V
save_4/Identity_36Identitysave_4/RestoreV2:35*
T0*
_output_shapes
:
V
save_4/AssignVariableOp_35AssignVariableOpslopesave_4/Identity_36*
dtype0
V
save_4/Identity_37Identitysave_4/RestoreV2:36*
T0*
_output_shapes
:
[
save_4/AssignVariableOp_36AssignVariableOp
slope/Adamsave_4/Identity_37*
dtype0
V
save_4/Identity_38Identitysave_4/RestoreV2:37*
T0*
_output_shapes
:
]
save_4/AssignVariableOp_37AssignVariableOpslope/Adam_1save_4/Identity_38*
dtype0
V
save_4/Identity_39Identitysave_4/RestoreV2:38*
T0*
_output_shapes
:
X
save_4/AssignVariableOp_38AssignVariableOpslope_1save_4/Identity_39*
dtype0
V
save_4/Identity_40Identitysave_4/RestoreV2:39*
T0*
_output_shapes
:
]
save_4/AssignVariableOp_39AssignVariableOpslope_1/Adamsave_4/Identity_40*
dtype0
V
save_4/Identity_41Identitysave_4/RestoreV2:40*
T0*
_output_shapes
:
_
save_4/AssignVariableOp_40AssignVariableOpslope_1/Adam_1save_4/Identity_41*
dtype0
V
save_4/Identity_42Identitysave_4/RestoreV2:41*
T0*
_output_shapes
:
X
save_4/AssignVariableOp_41AssignVariableOpslope_2save_4/Identity_42*
dtype0
V
save_4/Identity_43Identitysave_4/RestoreV2:42*
T0*
_output_shapes
:
]
save_4/AssignVariableOp_42AssignVariableOpslope_2/Adamsave_4/Identity_43*
dtype0
V
save_4/Identity_44Identitysave_4/RestoreV2:43*
T0*
_output_shapes
:
_
save_4/AssignVariableOp_43AssignVariableOpslope_2/Adam_1save_4/Identity_44*
dtype0
V
save_4/Identity_45Identitysave_4/RestoreV2:44*
T0*
_output_shapes
:
X
save_4/AssignVariableOp_44AssignVariableOpslope_3save_4/Identity_45*
dtype0
V
save_4/Identity_46Identitysave_4/RestoreV2:45*
T0*
_output_shapes
:
]
save_4/AssignVariableOp_45AssignVariableOpslope_3/Adamsave_4/Identity_46*
dtype0
V
save_4/Identity_47Identitysave_4/RestoreV2:46*
T0*
_output_shapes
:
_
save_4/AssignVariableOp_46AssignVariableOpslope_3/Adam_1save_4/Identity_47*
dtype0
V
save_4/Identity_48Identitysave_4/RestoreV2:47*
T0*
_output_shapes
:
X
save_4/AssignVariableOp_47AssignVariableOpslope_4save_4/Identity_48*
dtype0
V
save_4/Identity_49Identitysave_4/RestoreV2:48*
T0*
_output_shapes
:
]
save_4/AssignVariableOp_48AssignVariableOpslope_4/Adamsave_4/Identity_49*
dtype0
V
save_4/Identity_50Identitysave_4/RestoreV2:49*
T0*
_output_shapes
:
_
save_4/AssignVariableOp_49AssignVariableOpslope_4/Adam_1save_4/Identity_50*
dtype0
V
save_4/Identity_51Identitysave_4/RestoreV2:50*
T0*
_output_shapes
:
X
save_4/AssignVariableOp_50AssignVariableOpslope_5save_4/Identity_51*
dtype0
V
save_4/Identity_52Identitysave_4/RestoreV2:51*
T0*
_output_shapes
:
]
save_4/AssignVariableOp_51AssignVariableOpslope_5/Adamsave_4/Identity_52*
dtype0
V
save_4/Identity_53Identitysave_4/RestoreV2:52*
T0*
_output_shapes
:
_
save_4/AssignVariableOp_52AssignVariableOpslope_5/Adam_1save_4/Identity_53*
dtype0
С
save_4/restore_shardNoOp^save_4/AssignVariableOp^save_4/AssignVariableOp_1^save_4/AssignVariableOp_10^save_4/AssignVariableOp_11^save_4/AssignVariableOp_12^save_4/AssignVariableOp_13^save_4/AssignVariableOp_14^save_4/AssignVariableOp_15^save_4/AssignVariableOp_16^save_4/AssignVariableOp_17^save_4/AssignVariableOp_18^save_4/AssignVariableOp_19^save_4/AssignVariableOp_2^save_4/AssignVariableOp_20^save_4/AssignVariableOp_21^save_4/AssignVariableOp_22^save_4/AssignVariableOp_23^save_4/AssignVariableOp_24^save_4/AssignVariableOp_25^save_4/AssignVariableOp_26^save_4/AssignVariableOp_27^save_4/AssignVariableOp_28^save_4/AssignVariableOp_29^save_4/AssignVariableOp_3^save_4/AssignVariableOp_30^save_4/AssignVariableOp_31^save_4/AssignVariableOp_32^save_4/AssignVariableOp_33^save_4/AssignVariableOp_34^save_4/AssignVariableOp_35^save_4/AssignVariableOp_36^save_4/AssignVariableOp_37^save_4/AssignVariableOp_38^save_4/AssignVariableOp_39^save_4/AssignVariableOp_4^save_4/AssignVariableOp_40^save_4/AssignVariableOp_41^save_4/AssignVariableOp_42^save_4/AssignVariableOp_43^save_4/AssignVariableOp_44^save_4/AssignVariableOp_45^save_4/AssignVariableOp_46^save_4/AssignVariableOp_47^save_4/AssignVariableOp_48^save_4/AssignVariableOp_49^save_4/AssignVariableOp_5^save_4/AssignVariableOp_50^save_4/AssignVariableOp_51^save_4/AssignVariableOp_52^save_4/AssignVariableOp_6^save_4/AssignVariableOp_7^save_4/AssignVariableOp_8^save_4/AssignVariableOp_9
1
save_4/restore_allNoOp^save_4/restore_shard
L
Abs_60Absconv2d/mul_2*
T0*&
_output_shapes
:
N
	add_169/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_169AddV2Abs_60	add_169/y*
T0*&
_output_shapes
:
H
Log_120Logadd_169*
T0*&
_output_shapes
:
M
Const_91Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_121LogConst_91*
T0*
_output_shapes
: 
X

truediv_60RealDivLog_120Log_121*
T0*&
_output_shapes
:
O
ReadVariableOp_96ReadVariableOpslope*
_output_shapes
: *
dtype0
^
mul_104MulReadVariableOp_96
truediv_60*
T0*&
_output_shapes
:
S
ReadVariableOp_97ReadVariableOp	intercept*
_output_shapes
: *
dtype0
]
add_170AddV2ReadVariableOp_97mul_104*
T0*&
_output_shapes
:
{
differentiable_round_48Roundadd_170*
T0*
_gradient_op_type
Identity*&
_output_shapes
:
a
Const_92Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Min_24Mindifferentiable_round_48Const_92*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_32/packedPackMin_24*
N*
T0*
_output_shapes
:*

axis 
I
Rank_32Const*
_output_shapes
: *
dtype0*
value	B :
P
range_32/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_32/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_32Rangerange_32/startRank_32range_32/delta*

Tidx0*
_output_shapes
:
V
Min_25/inputPackMin_24*
N*
T0*
_output_shapes
:*

axis 
c
Min_25MinMin_25/inputrange_32*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
L
Abs_61Absconv2d/mul_2*
T0*&
_output_shapes
:
N
	add_171/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_171AddV2Abs_61	add_171/y*
T0*&
_output_shapes
:
H
Log_122Logadd_171*
T0*&
_output_shapes
:
M
Const_93Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_123LogConst_93*
T0*
_output_shapes
: 
X

truediv_61RealDivLog_122Log_123*
T0*&
_output_shapes
:
O
ReadVariableOp_98ReadVariableOpslope*
_output_shapes
: *
dtype0
^
mul_105MulReadVariableOp_98
truediv_61*
T0*&
_output_shapes
:
S
ReadVariableOp_99ReadVariableOp	intercept*
_output_shapes
: *
dtype0
]
add_172AddV2ReadVariableOp_99mul_105*
T0*&
_output_shapes
:
{
differentiable_round_49Roundadd_172*
T0*
_gradient_op_type
Identity*&
_output_shapes
:
a
Const_94Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Max_24Maxdifferentiable_round_49Const_94*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_33/packedPackMax_24*
N*
T0*
_output_shapes
:*

axis 
I
Rank_33Const*
_output_shapes
: *
dtype0*
value	B :
P
range_33/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_33/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_33Rangerange_33/startRank_33range_33/delta*

Tidx0*
_output_shapes
:
V
Max_25/inputPackMax_24*
N*
T0*
_output_shapes
:*

axis 
c
Max_25MaxMax_25/inputrange_33*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
>
sub_32SubMax_25Min_25*
T0*
_output_shapes
: 
N
	add_173/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
D
add_173AddV2sub_32	add_173/y*
T0*
_output_shapes
: 
7
Abs_62Absadd_173*
T0*
_output_shapes
: 
N
	add_174/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
D
add_174AddV2Abs_62	add_174/y*
T0*
_output_shapes
: 
8
Log_124Logadd_174*
T0*
_output_shapes
: 
M
Const_95Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_125LogConst_95*
T0*
_output_shapes
: 
H

truediv_62RealDivLog_124Log_125*
T0*
_output_shapes
: 
l
differentiable_ceil_12Ceil
truediv_62*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
N
	add_175/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
T
add_175AddV2	add_175/xdifferentiable_ceil_12*
T0*
_output_shapes
: 
N
Abs_63Absconv2d_1/mul_2*
T0*&
_output_shapes
:

N
	add_176/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_176AddV2Abs_63	add_176/y*
T0*&
_output_shapes
:

H
Log_126Logadd_176*
T0*&
_output_shapes
:

M
Const_96Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_127LogConst_96*
T0*
_output_shapes
: 
X

truediv_63RealDivLog_126Log_127*
T0*&
_output_shapes
:

R
ReadVariableOp_100ReadVariableOpslope_1*
_output_shapes
: *
dtype0
_
mul_106MulReadVariableOp_100
truediv_63*
T0*&
_output_shapes
:

V
ReadVariableOp_101ReadVariableOpintercept_1*
_output_shapes
: *
dtype0
^
add_177AddV2ReadVariableOp_101mul_106*
T0*&
_output_shapes
:

{
differentiable_round_50Roundadd_177*
T0*
_gradient_op_type
Identity*&
_output_shapes
:

a
Const_97Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Min_26Mindifferentiable_round_50Const_97*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_34/packedPackMin_26*
N*
T0*
_output_shapes
:*

axis 
I
Rank_34Const*
_output_shapes
: *
dtype0*
value	B :
P
range_34/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_34/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_34Rangerange_34/startRank_34range_34/delta*

Tidx0*
_output_shapes
:
V
Min_27/inputPackMin_26*
N*
T0*
_output_shapes
:*

axis 
c
Min_27MinMin_27/inputrange_34*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
N
Abs_64Absconv2d_1/mul_2*
T0*&
_output_shapes
:

N
	add_178/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
T
add_178AddV2Abs_64	add_178/y*
T0*&
_output_shapes
:

H
Log_128Logadd_178*
T0*&
_output_shapes
:

M
Const_98Const*
_output_shapes
: *
dtype0*
valueB
 *  @
9
Log_129LogConst_98*
T0*
_output_shapes
: 
X

truediv_64RealDivLog_128Log_129*
T0*&
_output_shapes
:

R
ReadVariableOp_102ReadVariableOpslope_1*
_output_shapes
: *
dtype0
_
mul_107MulReadVariableOp_102
truediv_64*
T0*&
_output_shapes
:

V
ReadVariableOp_103ReadVariableOpintercept_1*
_output_shapes
: *
dtype0
^
add_179AddV2ReadVariableOp_103mul_107*
T0*&
_output_shapes
:

{
differentiable_round_51Roundadd_179*
T0*
_gradient_op_type
Identity*&
_output_shapes
:

a
Const_99Const*
_output_shapes
:*
dtype0*%
valueB"             
n
Max_26Maxdifferentiable_round_51Const_99*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_35/packedPackMax_26*
N*
T0*
_output_shapes
:*

axis 
I
Rank_35Const*
_output_shapes
: *
dtype0*
value	B :
P
range_35/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_35/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_35Rangerange_35/startRank_35range_35/delta*

Tidx0*
_output_shapes
:
V
Max_27/inputPackMax_26*
N*
T0*
_output_shapes
:*

axis 
c
Max_27MaxMax_27/inputrange_35*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
>
sub_33SubMax_27Min_27*
T0*
_output_shapes
: 
N
	add_180/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
D
add_180AddV2sub_33	add_180/y*
T0*
_output_shapes
: 
7
Abs_65Absadd_180*
T0*
_output_shapes
: 
N
	add_181/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
D
add_181AddV2Abs_65	add_181/y*
T0*
_output_shapes
: 
8
Log_130Logadd_181*
T0*
_output_shapes
: 
N
	Const_100Const*
_output_shapes
: *
dtype0*
valueB
 *  @
:
Log_131Log	Const_100*
T0*
_output_shapes
: 
H

truediv_65RealDivLog_130Log_131*
T0*
_output_shapes
: 
l
differentiable_ceil_13Ceil
truediv_65*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
N
	add_182/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
T
add_182AddV2	add_182/xdifferentiable_ceil_13*
T0*
_output_shapes
: 
O
Abs_66Absconv2d_2/mul_2*
T0*'
_output_shapes
:1
А
N
	add_183/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
U
add_183AddV2Abs_66	add_183/y*
T0*'
_output_shapes
:1
А
I
Log_132Logadd_183*
T0*'
_output_shapes
:1
А
N
	Const_101Const*
_output_shapes
: *
dtype0*
valueB
 *  @
:
Log_133Log	Const_101*
T0*
_output_shapes
: 
Y

truediv_66RealDivLog_132Log_133*
T0*'
_output_shapes
:1
А
R
ReadVariableOp_104ReadVariableOpslope_2*
_output_shapes
: *
dtype0
`
mul_108MulReadVariableOp_104
truediv_66*
T0*'
_output_shapes
:1
А
V
ReadVariableOp_105ReadVariableOpintercept_2*
_output_shapes
: *
dtype0
_
add_184AddV2ReadVariableOp_105mul_108*
T0*'
_output_shapes
:1
А
|
differentiable_round_52Roundadd_184*
T0*
_gradient_op_type
Identity*'
_output_shapes
:1
А
b
	Const_102Const*
_output_shapes
:*
dtype0*%
valueB"             
o
Min_28Mindifferentiable_round_52	Const_102*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_36/packedPackMin_28*
N*
T0*
_output_shapes
:*

axis 
I
Rank_36Const*
_output_shapes
: *
dtype0*
value	B :
P
range_36/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_36/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_36Rangerange_36/startRank_36range_36/delta*

Tidx0*
_output_shapes
:
V
Min_29/inputPackMin_28*
N*
T0*
_output_shapes
:*

axis 
c
Min_29MinMin_29/inputrange_36*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
O
Abs_67Absconv2d_2/mul_2*
T0*'
_output_shapes
:1
А
N
	add_185/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
U
add_185AddV2Abs_67	add_185/y*
T0*'
_output_shapes
:1
А
I
Log_134Logadd_185*
T0*'
_output_shapes
:1
А
N
	Const_103Const*
_output_shapes
: *
dtype0*
valueB
 *  @
:
Log_135Log	Const_103*
T0*
_output_shapes
: 
Y

truediv_67RealDivLog_134Log_135*
T0*'
_output_shapes
:1
А
R
ReadVariableOp_106ReadVariableOpslope_2*
_output_shapes
: *
dtype0
`
mul_109MulReadVariableOp_106
truediv_67*
T0*'
_output_shapes
:1
А
V
ReadVariableOp_107ReadVariableOpintercept_2*
_output_shapes
: *
dtype0
_
add_186AddV2ReadVariableOp_107mul_109*
T0*'
_output_shapes
:1
А
|
differentiable_round_53Roundadd_186*
T0*
_gradient_op_type
Identity*'
_output_shapes
:1
А
b
	Const_104Const*
_output_shapes
:*
dtype0*%
valueB"             
o
Max_28Maxdifferentiable_round_53	Const_104*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_37/packedPackMax_28*
N*
T0*
_output_shapes
:*

axis 
I
Rank_37Const*
_output_shapes
: *
dtype0*
value	B :
P
range_37/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_37/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_37Rangerange_37/startRank_37range_37/delta*

Tidx0*
_output_shapes
:
V
Max_29/inputPackMax_28*
N*
T0*
_output_shapes
:*

axis 
c
Max_29MaxMax_29/inputrange_37*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
>
sub_34SubMax_29Min_29*
T0*
_output_shapes
: 
N
	add_187/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
D
add_187AddV2sub_34	add_187/y*
T0*
_output_shapes
: 
7
Abs_68Absadd_187*
T0*
_output_shapes
: 
N
	add_188/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
D
add_188AddV2Abs_68	add_188/y*
T0*
_output_shapes
: 
8
Log_136Logadd_188*
T0*
_output_shapes
: 
N
	Const_105Const*
_output_shapes
: *
dtype0*
valueB
 *  @
:
Log_137Log	Const_105*
T0*
_output_shapes
: 
H

truediv_68RealDivLog_136Log_137*
T0*
_output_shapes
: 
l
differentiable_ceil_14Ceil
truediv_68*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
N
	add_189/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
T
add_189AddV2	add_189/xdifferentiable_ceil_14*
T0*
_output_shapes
: 
Y
Abs_69Absconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
N
	add_190/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
^
add_190AddV2Abs_69	add_190/y*
T0*0
_output_shapes
:         А
R
Log_138Logadd_190*
T0*0
_output_shapes
:         А
N
	Const_106Const*
_output_shapes
: *
dtype0*
valueB
 *  @
:
Log_139Log	Const_106*
T0*
_output_shapes
: 
b

truediv_69RealDivLog_138Log_139*
T0*0
_output_shapes
:         А
R
ReadVariableOp_108ReadVariableOpslope_3*
_output_shapes
: *
dtype0
i
mul_110MulReadVariableOp_108
truediv_69*
T0*0
_output_shapes
:         А
V
ReadVariableOp_109ReadVariableOpintercept_3*
_output_shapes
: *
dtype0
h
add_191AddV2ReadVariableOp_109mul_110*
T0*0
_output_shapes
:         А
Е
differentiable_round_54Roundadd_191*
T0*
_gradient_op_type
Identity*0
_output_shapes
:         А
b
	Const_107Const*
_output_shapes
:*
dtype0*%
valueB"             
o
Min_30Mindifferentiable_round_54	Const_107*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_38/packedPackMin_30*
N*
T0*
_output_shapes
:*

axis 
I
Rank_38Const*
_output_shapes
: *
dtype0*
value	B :
P
range_38/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_38/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_38Rangerange_38/startRank_38range_38/delta*

Tidx0*
_output_shapes
:
V
Min_31/inputPackMin_30*
N*
T0*
_output_shapes
:*

axis 
c
Min_31MinMin_31/inputrange_38*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Y
Abs_70Absconv2d_2/Conv2D*
T0*0
_output_shapes
:         А
N
	add_192/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
^
add_192AddV2Abs_70	add_192/y*
T0*0
_output_shapes
:         А
R
Log_140Logadd_192*
T0*0
_output_shapes
:         А
N
	Const_108Const*
_output_shapes
: *
dtype0*
valueB
 *  @
:
Log_141Log	Const_108*
T0*
_output_shapes
: 
b

truediv_70RealDivLog_140Log_141*
T0*0
_output_shapes
:         А
R
ReadVariableOp_110ReadVariableOpslope_3*
_output_shapes
: *
dtype0
i
mul_111MulReadVariableOp_110
truediv_70*
T0*0
_output_shapes
:         А
V
ReadVariableOp_111ReadVariableOpintercept_3*
_output_shapes
: *
dtype0
h
add_193AddV2ReadVariableOp_111mul_111*
T0*0
_output_shapes
:         А
Е
differentiable_round_55Roundadd_193*
T0*
_gradient_op_type
Identity*0
_output_shapes
:         А
b
	Const_109Const*
_output_shapes
:*
dtype0*%
valueB"             
o
Max_30Maxdifferentiable_round_55	Const_109*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_39/packedPackMax_30*
N*
T0*
_output_shapes
:*

axis 
I
Rank_39Const*
_output_shapes
: *
dtype0*
value	B :
P
range_39/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_39/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_39Rangerange_39/startRank_39range_39/delta*

Tidx0*
_output_shapes
:
V
Max_31/inputPackMax_30*
N*
T0*
_output_shapes
:*

axis 
c
Max_31MaxMax_31/inputrange_39*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
>
sub_35SubMax_31Min_31*
T0*
_output_shapes
: 
N
	add_194/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
D
add_194AddV2sub_35	add_194/y*
T0*
_output_shapes
: 
7
Abs_71Absadd_194*
T0*
_output_shapes
: 
N
	add_195/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
D
add_195AddV2Abs_71	add_195/y*
T0*
_output_shapes
: 
8
Log_142Logadd_195*
T0*
_output_shapes
: 
N
	Const_110Const*
_output_shapes
: *
dtype0*
valueB
 *  @
:
Log_143Log	Const_110*
T0*
_output_shapes
: 
H

truediv_71RealDivLog_142Log_143*
T0*
_output_shapes
: 
l
differentiable_ceil_15Ceil
truediv_71*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
N
	add_196/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
T
add_196AddV2	add_196/xdifferentiable_ceil_15*
T0*
_output_shapes
: 
P
Abs_72Absconv2d_3/mul_2*
T0*(
_output_shapes
:АА
N
	add_197/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
V
add_197AddV2Abs_72	add_197/y*
T0*(
_output_shapes
:АА
J
Log_144Logadd_197*
T0*(
_output_shapes
:АА
N
	Const_111Const*
_output_shapes
: *
dtype0*
valueB
 *  @
:
Log_145Log	Const_111*
T0*
_output_shapes
: 
Z

truediv_72RealDivLog_144Log_145*
T0*(
_output_shapes
:АА
R
ReadVariableOp_112ReadVariableOpslope_4*
_output_shapes
: *
dtype0
a
mul_112MulReadVariableOp_112
truediv_72*
T0*(
_output_shapes
:АА
V
ReadVariableOp_113ReadVariableOpintercept_4*
_output_shapes
: *
dtype0
`
add_198AddV2ReadVariableOp_113mul_112*
T0*(
_output_shapes
:АА
}
differentiable_round_56Roundadd_198*
T0*
_gradient_op_type
Identity*(
_output_shapes
:АА
b
	Const_112Const*
_output_shapes
:*
dtype0*%
valueB"             
o
Min_32Mindifferentiable_round_56	Const_112*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_40/packedPackMin_32*
N*
T0*
_output_shapes
:*

axis 
I
Rank_40Const*
_output_shapes
: *
dtype0*
value	B :
P
range_40/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_40/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_40Rangerange_40/startRank_40range_40/delta*

Tidx0*
_output_shapes
:
V
Min_33/inputPackMin_32*
N*
T0*
_output_shapes
:*

axis 
c
Min_33MinMin_33/inputrange_40*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
P
Abs_73Absconv2d_3/mul_2*
T0*(
_output_shapes
:АА
N
	add_199/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
V
add_199AddV2Abs_73	add_199/y*
T0*(
_output_shapes
:АА
J
Log_146Logadd_199*
T0*(
_output_shapes
:АА
N
	Const_113Const*
_output_shapes
: *
dtype0*
valueB
 *  @
:
Log_147Log	Const_113*
T0*
_output_shapes
: 
Z

truediv_73RealDivLog_146Log_147*
T0*(
_output_shapes
:АА
R
ReadVariableOp_114ReadVariableOpslope_4*
_output_shapes
: *
dtype0
a
mul_113MulReadVariableOp_114
truediv_73*
T0*(
_output_shapes
:АА
V
ReadVariableOp_115ReadVariableOpintercept_4*
_output_shapes
: *
dtype0
`
add_200AddV2ReadVariableOp_115mul_113*
T0*(
_output_shapes
:АА
}
differentiable_round_57Roundadd_200*
T0*
_gradient_op_type
Identity*(
_output_shapes
:АА
b
	Const_114Const*
_output_shapes
:*
dtype0*%
valueB"             
o
Max_32Maxdifferentiable_round_57	Const_114*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_41/packedPackMax_32*
N*
T0*
_output_shapes
:*

axis 
I
Rank_41Const*
_output_shapes
: *
dtype0*
value	B :
P
range_41/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_41/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_41Rangerange_41/startRank_41range_41/delta*

Tidx0*
_output_shapes
:
V
Max_33/inputPackMax_32*
N*
T0*
_output_shapes
:*

axis 
c
Max_33MaxMax_33/inputrange_41*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
>
sub_36SubMax_33Min_33*
T0*
_output_shapes
: 
N
	add_201/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
D
add_201AddV2sub_36	add_201/y*
T0*
_output_shapes
: 
7
Abs_74Absadd_201*
T0*
_output_shapes
: 
N
	add_202/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
D
add_202AddV2Abs_74	add_202/y*
T0*
_output_shapes
: 
8
Log_148Logadd_202*
T0*
_output_shapes
: 
N
	Const_115Const*
_output_shapes
: *
dtype0*
valueB
 *  @
:
Log_149Log	Const_115*
T0*
_output_shapes
: 
H

truediv_74RealDivLog_148Log_149*
T0*
_output_shapes
: 
l
differentiable_ceil_16Ceil
truediv_74*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
N
	add_203/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
T
add_203AddV2	add_203/xdifferentiable_ceil_16*
T0*
_output_shapes
: 
O
Abs_75Absconv2d_4/mul_2*
T0*'
_output_shapes
:А

N
	add_204/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
U
add_204AddV2Abs_75	add_204/y*
T0*'
_output_shapes
:А

I
Log_150Logadd_204*
T0*'
_output_shapes
:А

N
	Const_116Const*
_output_shapes
: *
dtype0*
valueB
 *  @
:
Log_151Log	Const_116*
T0*
_output_shapes
: 
Y

truediv_75RealDivLog_150Log_151*
T0*'
_output_shapes
:А

R
ReadVariableOp_116ReadVariableOpslope_5*
_output_shapes
: *
dtype0
`
mul_114MulReadVariableOp_116
truediv_75*
T0*'
_output_shapes
:А

V
ReadVariableOp_117ReadVariableOpintercept_5*
_output_shapes
: *
dtype0
_
add_205AddV2ReadVariableOp_117mul_114*
T0*'
_output_shapes
:А

|
differentiable_round_58Roundadd_205*
T0*
_gradient_op_type
Identity*'
_output_shapes
:А

b
	Const_117Const*
_output_shapes
:*
dtype0*%
valueB"             
o
Min_34Mindifferentiable_round_58	Const_117*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_42/packedPackMin_34*
N*
T0*
_output_shapes
:*

axis 
I
Rank_42Const*
_output_shapes
: *
dtype0*
value	B :
P
range_42/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_42/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_42Rangerange_42/startRank_42range_42/delta*

Tidx0*
_output_shapes
:
V
Min_35/inputPackMin_34*
N*
T0*
_output_shapes
:*

axis 
c
Min_35MinMin_35/inputrange_42*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
O
Abs_76Absconv2d_4/mul_2*
T0*'
_output_shapes
:А

N
	add_206/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
U
add_206AddV2Abs_76	add_206/y*
T0*'
_output_shapes
:А

I
Log_152Logadd_206*
T0*'
_output_shapes
:А

N
	Const_118Const*
_output_shapes
: *
dtype0*
valueB
 *  @
:
Log_153Log	Const_118*
T0*
_output_shapes
: 
Y

truediv_76RealDivLog_152Log_153*
T0*'
_output_shapes
:А

R
ReadVariableOp_118ReadVariableOpslope_5*
_output_shapes
: *
dtype0
`
mul_115MulReadVariableOp_118
truediv_76*
T0*'
_output_shapes
:А

V
ReadVariableOp_119ReadVariableOpintercept_5*
_output_shapes
: *
dtype0
_
add_207AddV2ReadVariableOp_119mul_115*
T0*'
_output_shapes
:А

|
differentiable_round_59Roundadd_207*
T0*
_gradient_op_type
Identity*'
_output_shapes
:А

b
	Const_119Const*
_output_shapes
:*
dtype0*%
valueB"             
o
Max_34Maxdifferentiable_round_59	Const_119*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
X
Rank_43/packedPackMax_34*
N*
T0*
_output_shapes
:*

axis 
I
Rank_43Const*
_output_shapes
: *
dtype0*
value	B :
P
range_43/startConst*
_output_shapes
: *
dtype0*
value	B : 
P
range_43/deltaConst*
_output_shapes
: *
dtype0*
value	B :
b
range_43Rangerange_43/startRank_43range_43/delta*

Tidx0*
_output_shapes
:
V
Max_35/inputPackMax_34*
N*
T0*
_output_shapes
:*

axis 
c
Max_35MaxMax_35/inputrange_43*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
>
sub_37SubMax_35Min_35*
T0*
_output_shapes
: 
N
	add_208/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
D
add_208AddV2sub_37	add_208/y*
T0*
_output_shapes
: 
7
Abs_77Absadd_208*
T0*
_output_shapes
: 
N
	add_209/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5
D
add_209AddV2Abs_77	add_209/y*
T0*
_output_shapes
: 
8
Log_154Logadd_209*
T0*
_output_shapes
: 
N
	Const_120Const*
_output_shapes
: *
dtype0*
valueB
 *  @
:
Log_155Log	Const_120*
T0*
_output_shapes
: 
H

truediv_77RealDivLog_154Log_155*
T0*
_output_shapes
: 
l
differentiable_ceil_17Ceil
truediv_77*
T0*
_gradient_op_type
Identity*
_output_shapes
: 
N
	add_210/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
T
add_210AddV2	add_210/xdifferentiable_ceil_17*
T0*
_output_shapes
: 
[
save_5/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
_output_shapes
: *
dtype0*
shape: 
Ж
save_5/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1f85a35338b0450c9b589ed6babc5471/part
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_5/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_5/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
Е
save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
├
save_5/SaveV2/tensor_namesConst*
_output_shapes
:5*
dtype0*Ї
valueъBч5Bbeta1_powerBbeta2_powerBconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1Bconv2d_2/kernelBconv2d_2/kernel/AdamBconv2d_2/kernel/Adam_1Bconv2d_3/kernelBconv2d_3/kernel/AdamBconv2d_3/kernel/Adam_1Bconv2d_4/kernelBconv2d_4/kernel/AdamBconv2d_4/kernel/Adam_1B	interceptBintercept/AdamBintercept/Adam_1Bintercept_1Bintercept_1/AdamBintercept_1/Adam_1Bintercept_2Bintercept_2/AdamBintercept_2/Adam_1Bintercept_3Bintercept_3/AdamBintercept_3/Adam_1Bintercept_4Bintercept_4/AdamBintercept_4/Adam_1Bintercept_5Bintercept_5/AdamBintercept_5/Adam_1BslopeB
slope/AdamBslope/Adam_1Bslope_1Bslope_1/AdamBslope_1/Adam_1Bslope_2Bslope_2/AdamBslope_2/Adam_1Bslope_3Bslope_3/AdamBslope_3/Adam_1Bslope_4Bslope_4/AdamBslope_4/Adam_1Bslope_5Bslope_5/AdamBslope_5/Adam_1
╧
save_5/SaveV2/shape_and_slicesConst*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
│
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_power/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOp&conv2d/kernel/Adam/Read/ReadVariableOp(conv2d/kernel/Adam_1/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp(conv2d_1/kernel/Adam/Read/ReadVariableOp*conv2d_1/kernel/Adam_1/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp(conv2d_2/kernel/Adam/Read/ReadVariableOp*conv2d_2/kernel/Adam_1/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp(conv2d_3/kernel/Adam/Read/ReadVariableOp*conv2d_3/kernel/Adam_1/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp(conv2d_4/kernel/Adam/Read/ReadVariableOp*conv2d_4/kernel/Adam_1/Read/ReadVariableOpintercept/Read/ReadVariableOp"intercept/Adam/Read/ReadVariableOp$intercept/Adam_1/Read/ReadVariableOpintercept_1/Read/ReadVariableOp$intercept_1/Adam/Read/ReadVariableOp&intercept_1/Adam_1/Read/ReadVariableOpintercept_2/Read/ReadVariableOp$intercept_2/Adam/Read/ReadVariableOp&intercept_2/Adam_1/Read/ReadVariableOpintercept_3/Read/ReadVariableOp$intercept_3/Adam/Read/ReadVariableOp&intercept_3/Adam_1/Read/ReadVariableOpintercept_4/Read/ReadVariableOp$intercept_4/Adam/Read/ReadVariableOp&intercept_4/Adam_1/Read/ReadVariableOpintercept_5/Read/ReadVariableOp$intercept_5/Adam/Read/ReadVariableOp&intercept_5/Adam_1/Read/ReadVariableOpslope/Read/ReadVariableOpslope/Adam/Read/ReadVariableOp slope/Adam_1/Read/ReadVariableOpslope_1/Read/ReadVariableOp slope_1/Adam/Read/ReadVariableOp"slope_1/Adam_1/Read/ReadVariableOpslope_2/Read/ReadVariableOp slope_2/Adam/Read/ReadVariableOp"slope_2/Adam_1/Read/ReadVariableOpslope_3/Read/ReadVariableOp slope_3/Adam/Read/ReadVariableOp"slope_3/Adam_1/Read/ReadVariableOpslope_4/Read/ReadVariableOp slope_4/Adam/Read/ReadVariableOp"slope_4/Adam_1/Read/ReadVariableOpslope_5/Read/ReadVariableOp slope_5/Adam/Read/ReadVariableOp"slope_5/Adam_1/Read/ReadVariableOp*C
dtypes9
725
Щ
save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*
T0*)
_class
loc:@save_5/ShardedFilename*
_output_shapes
: 
г
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
N*
T0*
_output_shapes
:*

axis 
Г
save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(
В
save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
T0*
_output_shapes
: 
╞
save_5/RestoreV2/tensor_namesConst*
_output_shapes
:5*
dtype0*Ї
valueъBч5Bbeta1_powerBbeta2_powerBconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1Bconv2d_2/kernelBconv2d_2/kernel/AdamBconv2d_2/kernel/Adam_1Bconv2d_3/kernelBconv2d_3/kernel/AdamBconv2d_3/kernel/Adam_1Bconv2d_4/kernelBconv2d_4/kernel/AdamBconv2d_4/kernel/Adam_1B	interceptBintercept/AdamBintercept/Adam_1Bintercept_1Bintercept_1/AdamBintercept_1/Adam_1Bintercept_2Bintercept_2/AdamBintercept_2/Adam_1Bintercept_3Bintercept_3/AdamBintercept_3/Adam_1Bintercept_4Bintercept_4/AdamBintercept_4/Adam_1Bintercept_5Bintercept_5/AdamBintercept_5/Adam_1BslopeB
slope/AdamBslope/Adam_1Bslope_1Bslope_1/AdamBslope_1/Adam_1Bslope_2Bslope_2/AdamBslope_2/Adam_1Bslope_3Bslope_3/AdamBslope_3/Adam_1Bslope_4Bslope_4/AdamBslope_4/Adam_1Bslope_5Bslope_5/AdamBslope_5/Adam_1
╥
!save_5/RestoreV2/shape_and_slicesConst*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Я
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*ъ
_output_shapes╫
╘:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725
R
save_5/Identity_1Identitysave_5/RestoreV2*
T0*
_output_shapes
:
X
save_5/AssignVariableOpAssignVariableOpbeta1_powersave_5/Identity_1*
dtype0
T
save_5/Identity_2Identitysave_5/RestoreV2:1*
T0*
_output_shapes
:
Z
save_5/AssignVariableOp_1AssignVariableOpbeta2_powersave_5/Identity_2*
dtype0
T
save_5/Identity_3Identitysave_5/RestoreV2:2*
T0*
_output_shapes
:
\
save_5/AssignVariableOp_2AssignVariableOpconv2d/kernelsave_5/Identity_3*
dtype0
T
save_5/Identity_4Identitysave_5/RestoreV2:3*
T0*
_output_shapes
:
a
save_5/AssignVariableOp_3AssignVariableOpconv2d/kernel/Adamsave_5/Identity_4*
dtype0
T
save_5/Identity_5Identitysave_5/RestoreV2:4*
T0*
_output_shapes
:
c
save_5/AssignVariableOp_4AssignVariableOpconv2d/kernel/Adam_1save_5/Identity_5*
dtype0
T
save_5/Identity_6Identitysave_5/RestoreV2:5*
T0*
_output_shapes
:
^
save_5/AssignVariableOp_5AssignVariableOpconv2d_1/kernelsave_5/Identity_6*
dtype0
T
save_5/Identity_7Identitysave_5/RestoreV2:6*
T0*
_output_shapes
:
c
save_5/AssignVariableOp_6AssignVariableOpconv2d_1/kernel/Adamsave_5/Identity_7*
dtype0
T
save_5/Identity_8Identitysave_5/RestoreV2:7*
T0*
_output_shapes
:
e
save_5/AssignVariableOp_7AssignVariableOpconv2d_1/kernel/Adam_1save_5/Identity_8*
dtype0
T
save_5/Identity_9Identitysave_5/RestoreV2:8*
T0*
_output_shapes
:
^
save_5/AssignVariableOp_8AssignVariableOpconv2d_2/kernelsave_5/Identity_9*
dtype0
U
save_5/Identity_10Identitysave_5/RestoreV2:9*
T0*
_output_shapes
:
d
save_5/AssignVariableOp_9AssignVariableOpconv2d_2/kernel/Adamsave_5/Identity_10*
dtype0
V
save_5/Identity_11Identitysave_5/RestoreV2:10*
T0*
_output_shapes
:
g
save_5/AssignVariableOp_10AssignVariableOpconv2d_2/kernel/Adam_1save_5/Identity_11*
dtype0
V
save_5/Identity_12Identitysave_5/RestoreV2:11*
T0*
_output_shapes
:
`
save_5/AssignVariableOp_11AssignVariableOpconv2d_3/kernelsave_5/Identity_12*
dtype0
V
save_5/Identity_13Identitysave_5/RestoreV2:12*
T0*
_output_shapes
:
e
save_5/AssignVariableOp_12AssignVariableOpconv2d_3/kernel/Adamsave_5/Identity_13*
dtype0
V
save_5/Identity_14Identitysave_5/RestoreV2:13*
T0*
_output_shapes
:
g
save_5/AssignVariableOp_13AssignVariableOpconv2d_3/kernel/Adam_1save_5/Identity_14*
dtype0
V
save_5/Identity_15Identitysave_5/RestoreV2:14*
T0*
_output_shapes
:
`
save_5/AssignVariableOp_14AssignVariableOpconv2d_4/kernelsave_5/Identity_15*
dtype0
V
save_5/Identity_16Identitysave_5/RestoreV2:15*
T0*
_output_shapes
:
e
save_5/AssignVariableOp_15AssignVariableOpconv2d_4/kernel/Adamsave_5/Identity_16*
dtype0
V
save_5/Identity_17Identitysave_5/RestoreV2:16*
T0*
_output_shapes
:
g
save_5/AssignVariableOp_16AssignVariableOpconv2d_4/kernel/Adam_1save_5/Identity_17*
dtype0
V
save_5/Identity_18Identitysave_5/RestoreV2:17*
T0*
_output_shapes
:
Z
save_5/AssignVariableOp_17AssignVariableOp	interceptsave_5/Identity_18*
dtype0
V
save_5/Identity_19Identitysave_5/RestoreV2:18*
T0*
_output_shapes
:
_
save_5/AssignVariableOp_18AssignVariableOpintercept/Adamsave_5/Identity_19*
dtype0
V
save_5/Identity_20Identitysave_5/RestoreV2:19*
T0*
_output_shapes
:
a
save_5/AssignVariableOp_19AssignVariableOpintercept/Adam_1save_5/Identity_20*
dtype0
V
save_5/Identity_21Identitysave_5/RestoreV2:20*
T0*
_output_shapes
:
\
save_5/AssignVariableOp_20AssignVariableOpintercept_1save_5/Identity_21*
dtype0
V
save_5/Identity_22Identitysave_5/RestoreV2:21*
T0*
_output_shapes
:
a
save_5/AssignVariableOp_21AssignVariableOpintercept_1/Adamsave_5/Identity_22*
dtype0
V
save_5/Identity_23Identitysave_5/RestoreV2:22*
T0*
_output_shapes
:
c
save_5/AssignVariableOp_22AssignVariableOpintercept_1/Adam_1save_5/Identity_23*
dtype0
V
save_5/Identity_24Identitysave_5/RestoreV2:23*
T0*
_output_shapes
:
\
save_5/AssignVariableOp_23AssignVariableOpintercept_2save_5/Identity_24*
dtype0
V
save_5/Identity_25Identitysave_5/RestoreV2:24*
T0*
_output_shapes
:
a
save_5/AssignVariableOp_24AssignVariableOpintercept_2/Adamsave_5/Identity_25*
dtype0
V
save_5/Identity_26Identitysave_5/RestoreV2:25*
T0*
_output_shapes
:
c
save_5/AssignVariableOp_25AssignVariableOpintercept_2/Adam_1save_5/Identity_26*
dtype0
V
save_5/Identity_27Identitysave_5/RestoreV2:26*
T0*
_output_shapes
:
\
save_5/AssignVariableOp_26AssignVariableOpintercept_3save_5/Identity_27*
dtype0
V
save_5/Identity_28Identitysave_5/RestoreV2:27*
T0*
_output_shapes
:
a
save_5/AssignVariableOp_27AssignVariableOpintercept_3/Adamsave_5/Identity_28*
dtype0
V
save_5/Identity_29Identitysave_5/RestoreV2:28*
T0*
_output_shapes
:
c
save_5/AssignVariableOp_28AssignVariableOpintercept_3/Adam_1save_5/Identity_29*
dtype0
V
save_5/Identity_30Identitysave_5/RestoreV2:29*
T0*
_output_shapes
:
\
save_5/AssignVariableOp_29AssignVariableOpintercept_4save_5/Identity_30*
dtype0
V
save_5/Identity_31Identitysave_5/RestoreV2:30*
T0*
_output_shapes
:
a
save_5/AssignVariableOp_30AssignVariableOpintercept_4/Adamsave_5/Identity_31*
dtype0
V
save_5/Identity_32Identitysave_5/RestoreV2:31*
T0*
_output_shapes
:
c
save_5/AssignVariableOp_31AssignVariableOpintercept_4/Adam_1save_5/Identity_32*
dtype0
V
save_5/Identity_33Identitysave_5/RestoreV2:32*
T0*
_output_shapes
:
\
save_5/AssignVariableOp_32AssignVariableOpintercept_5save_5/Identity_33*
dtype0
V
save_5/Identity_34Identitysave_5/RestoreV2:33*
T0*
_output_shapes
:
a
save_5/AssignVariableOp_33AssignVariableOpintercept_5/Adamsave_5/Identity_34*
dtype0
V
save_5/Identity_35Identitysave_5/RestoreV2:34*
T0*
_output_shapes
:
c
save_5/AssignVariableOp_34AssignVariableOpintercept_5/Adam_1save_5/Identity_35*
dtype0
V
save_5/Identity_36Identitysave_5/RestoreV2:35*
T0*
_output_shapes
:
V
save_5/AssignVariableOp_35AssignVariableOpslopesave_5/Identity_36*
dtype0
V
save_5/Identity_37Identitysave_5/RestoreV2:36*
T0*
_output_shapes
:
[
save_5/AssignVariableOp_36AssignVariableOp
slope/Adamsave_5/Identity_37*
dtype0
V
save_5/Identity_38Identitysave_5/RestoreV2:37*
T0*
_output_shapes
:
]
save_5/AssignVariableOp_37AssignVariableOpslope/Adam_1save_5/Identity_38*
dtype0
V
save_5/Identity_39Identitysave_5/RestoreV2:38*
T0*
_output_shapes
:
X
save_5/AssignVariableOp_38AssignVariableOpslope_1save_5/Identity_39*
dtype0
V
save_5/Identity_40Identitysave_5/RestoreV2:39*
T0*
_output_shapes
:
]
save_5/AssignVariableOp_39AssignVariableOpslope_1/Adamsave_5/Identity_40*
dtype0
V
save_5/Identity_41Identitysave_5/RestoreV2:40*
T0*
_output_shapes
:
_
save_5/AssignVariableOp_40AssignVariableOpslope_1/Adam_1save_5/Identity_41*
dtype0
V
save_5/Identity_42Identitysave_5/RestoreV2:41*
T0*
_output_shapes
:
X
save_5/AssignVariableOp_41AssignVariableOpslope_2save_5/Identity_42*
dtype0
V
save_5/Identity_43Identitysave_5/RestoreV2:42*
T0*
_output_shapes
:
]
save_5/AssignVariableOp_42AssignVariableOpslope_2/Adamsave_5/Identity_43*
dtype0
V
save_5/Identity_44Identitysave_5/RestoreV2:43*
T0*
_output_shapes
:
_
save_5/AssignVariableOp_43AssignVariableOpslope_2/Adam_1save_5/Identity_44*
dtype0
V
save_5/Identity_45Identitysave_5/RestoreV2:44*
T0*
_output_shapes
:
X
save_5/AssignVariableOp_44AssignVariableOpslope_3save_5/Identity_45*
dtype0
V
save_5/Identity_46Identitysave_5/RestoreV2:45*
T0*
_output_shapes
:
]
save_5/AssignVariableOp_45AssignVariableOpslope_3/Adamsave_5/Identity_46*
dtype0
V
save_5/Identity_47Identitysave_5/RestoreV2:46*
T0*
_output_shapes
:
_
save_5/AssignVariableOp_46AssignVariableOpslope_3/Adam_1save_5/Identity_47*
dtype0
V
save_5/Identity_48Identitysave_5/RestoreV2:47*
T0*
_output_shapes
:
X
save_5/AssignVariableOp_47AssignVariableOpslope_4save_5/Identity_48*
dtype0
V
save_5/Identity_49Identitysave_5/RestoreV2:48*
T0*
_output_shapes
:
]
save_5/AssignVariableOp_48AssignVariableOpslope_4/Adamsave_5/Identity_49*
dtype0
V
save_5/Identity_50Identitysave_5/RestoreV2:49*
T0*
_output_shapes
:
_
save_5/AssignVariableOp_49AssignVariableOpslope_4/Adam_1save_5/Identity_50*
dtype0
V
save_5/Identity_51Identitysave_5/RestoreV2:50*
T0*
_output_shapes
:
X
save_5/AssignVariableOp_50AssignVariableOpslope_5save_5/Identity_51*
dtype0
V
save_5/Identity_52Identitysave_5/RestoreV2:51*
T0*
_output_shapes
:
]
save_5/AssignVariableOp_51AssignVariableOpslope_5/Adamsave_5/Identity_52*
dtype0
V
save_5/Identity_53Identitysave_5/RestoreV2:52*
T0*
_output_shapes
:
_
save_5/AssignVariableOp_52AssignVariableOpslope_5/Adam_1save_5/Identity_53*
dtype0
С
save_5/restore_shardNoOp^save_5/AssignVariableOp^save_5/AssignVariableOp_1^save_5/AssignVariableOp_10^save_5/AssignVariableOp_11^save_5/AssignVariableOp_12^save_5/AssignVariableOp_13^save_5/AssignVariableOp_14^save_5/AssignVariableOp_15^save_5/AssignVariableOp_16^save_5/AssignVariableOp_17^save_5/AssignVariableOp_18^save_5/AssignVariableOp_19^save_5/AssignVariableOp_2^save_5/AssignVariableOp_20^save_5/AssignVariableOp_21^save_5/AssignVariableOp_22^save_5/AssignVariableOp_23^save_5/AssignVariableOp_24^save_5/AssignVariableOp_25^save_5/AssignVariableOp_26^save_5/AssignVariableOp_27^save_5/AssignVariableOp_28^save_5/AssignVariableOp_29^save_5/AssignVariableOp_3^save_5/AssignVariableOp_30^save_5/AssignVariableOp_31^save_5/AssignVariableOp_32^save_5/AssignVariableOp_33^save_5/AssignVariableOp_34^save_5/AssignVariableOp_35^save_5/AssignVariableOp_36^save_5/AssignVariableOp_37^save_5/AssignVariableOp_38^save_5/AssignVariableOp_39^save_5/AssignVariableOp_4^save_5/AssignVariableOp_40^save_5/AssignVariableOp_41^save_5/AssignVariableOp_42^save_5/AssignVariableOp_43^save_5/AssignVariableOp_44^save_5/AssignVariableOp_45^save_5/AssignVariableOp_46^save_5/AssignVariableOp_47^save_5/AssignVariableOp_48^save_5/AssignVariableOp_49^save_5/AssignVariableOp_5^save_5/AssignVariableOp_50^save_5/AssignVariableOp_51^save_5/AssignVariableOp_52^save_5/AssignVariableOp_6^save_5/AssignVariableOp_7^save_5/AssignVariableOp_8^save_5/AssignVariableOp_9
1
save_5/restore_allNoOp^save_5/restore_shard "пB
save_5/Const:0save_5/Identity:0save_5/restore_all (5 @F8"▒	
	summariesг	
а	
hp_cross_entropy_flatten_1:0
distillation_loss_flatten_1:0
b_i_t___l_o_s_s:0
'r_e_g_u_l_a_r_i_z_a_t_i_o_n___t_e_r_m:0
lp_accuracy_flatten_1:0
hp_accuracy_flatten_1:0
total_loss_total_loss:0
weights_bits_conv2d:0
weights_bits_conv2d_1:0
weights_bits_conv2d_2:0
weights_bits_activation_3:0
weights_bits_conv2d_3:0
weights_bits_conv2d_4:0
bits_average:0
qerr_op_activation:0
qerr_op_activation_1:0
qerr_op_activation_2:0
qerr_op_activation_3:0
qerr_op_activation_4:0
qerr_op_average:0
qerr_weights_bias_conv2d:0
qerr_weights_bias_conv2d_1:0
qerr_weights_bias_conv2d_2:0
qerr_weights_bias_conv2d_3:0
qerr_weights_bias_conv2d_4:0
qerr_weights_weights_conv2d:0
qerr_weights_weights_conv2d_1:0
qerr_weights_weights_conv2d_2:0
qerr_weights_weights_conv2d_3:0
qerr_weights_weights_conv2d_4:0
qerr_weights_average:0
qangle_op_activation:0
qangle_op_activation_1:0
qangle_op_activation_2:0
qangle_op_activation_3:0
qangle_op_activation_4:0
qangle_op_average:0
qangle_weights_weights_conv2d:0
!qangle_weights_weights_conv2d_1:0
!qangle_weights_weights_conv2d_2:0
!qangle_weights_weights_conv2d_3:0
!qangle_weights_weights_conv2d_4:0
qangle_weights_average:0"
train_op

Adam"╦
trainable_variables│░
k
intercept:0intercept/Assignintercept/Read/ReadVariableOp:0(2%intercept/Initializer/initial_value:08
[
slope:0slope/Assignslope/Read/ReadVariableOp:0(2!slope/Initializer/initial_value:08
s
intercept_1:0intercept_1/Assign!intercept_1/Read/ReadVariableOp:0(2'intercept_1/Initializer/initial_value:08
c
	slope_1:0slope_1/Assignslope_1/Read/ReadVariableOp:0(2#slope_1/Initializer/initial_value:08
s
intercept_2:0intercept_2/Assign!intercept_2/Read/ReadVariableOp:0(2'intercept_2/Initializer/initial_value:08
c
	slope_2:0slope_2/Assignslope_2/Read/ReadVariableOp:0(2#slope_2/Initializer/initial_value:08
s
intercept_3:0intercept_3/Assign!intercept_3/Read/ReadVariableOp:0(2'intercept_3/Initializer/initial_value:08
c
	slope_3:0slope_3/Assignslope_3/Read/ReadVariableOp:0(2#slope_3/Initializer/initial_value:08
s
intercept_4:0intercept_4/Assign!intercept_4/Read/ReadVariableOp:0(2'intercept_4/Initializer/initial_value:08
c
	slope_4:0slope_4/Assignslope_4/Read/ReadVariableOp:0(2#slope_4/Initializer/initial_value:08
s
intercept_5:0intercept_5/Assign!intercept_5/Read/ReadVariableOp:0(2'intercept_5/Initializer/initial_value:08
c
	slope_5:0slope_5/Assignslope_5/Read/ReadVariableOp:0(2#slope_5/Initializer/initial_value:08
~
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2,conv2d/kernel/Initializer/truncated_normal:08
Ж
conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2.conv2d_1/kernel/Initializer/truncated_normal:08
Ж
conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2.conv2d_2/kernel/Initializer/truncated_normal:08
Ж
conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2.conv2d_3/kernel/Initializer/truncated_normal:08
Ж
conv2d_4/kernel:0conv2d_4/kernel/Assign%conv2d_4/kernel/Read/ReadVariableOp:0(2.conv2d_4/kernel/Initializer/truncated_normal:08"╠3
	variables╛3╗3
k
intercept:0intercept/Assignintercept/Read/ReadVariableOp:0(2%intercept/Initializer/initial_value:08
[
slope:0slope/Assignslope/Read/ReadVariableOp:0(2!slope/Initializer/initial_value:08
s
intercept_1:0intercept_1/Assign!intercept_1/Read/ReadVariableOp:0(2'intercept_1/Initializer/initial_value:08
c
	slope_1:0slope_1/Assignslope_1/Read/ReadVariableOp:0(2#slope_1/Initializer/initial_value:08
s
intercept_2:0intercept_2/Assign!intercept_2/Read/ReadVariableOp:0(2'intercept_2/Initializer/initial_value:08
c
	slope_2:0slope_2/Assignslope_2/Read/ReadVariableOp:0(2#slope_2/Initializer/initial_value:08
s
intercept_3:0intercept_3/Assign!intercept_3/Read/ReadVariableOp:0(2'intercept_3/Initializer/initial_value:08
c
	slope_3:0slope_3/Assignslope_3/Read/ReadVariableOp:0(2#slope_3/Initializer/initial_value:08
s
intercept_4:0intercept_4/Assign!intercept_4/Read/ReadVariableOp:0(2'intercept_4/Initializer/initial_value:08
c
	slope_4:0slope_4/Assignslope_4/Read/ReadVariableOp:0(2#slope_4/Initializer/initial_value:08
s
intercept_5:0intercept_5/Assign!intercept_5/Read/ReadVariableOp:0(2'intercept_5/Initializer/initial_value:08
c
	slope_5:0slope_5/Assignslope_5/Read/ReadVariableOp:0(2#slope_5/Initializer/initial_value:08
~
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2,conv2d/kernel/Initializer/truncated_normal:08
Ж
conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2.conv2d_1/kernel/Initializer/truncated_normal:08
Ж
conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2.conv2d_2/kernel/Initializer/truncated_normal:08
Ж
conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2.conv2d_3/kernel/Initializer/truncated_normal:08
Ж
conv2d_4/kernel:0conv2d_4/kernel/Assign%conv2d_4/kernel/Read/ReadVariableOp:0(2.conv2d_4/kernel/Initializer/truncated_normal:08
q
beta1_power:0beta1_power/Assign!beta1_power/Read/ReadVariableOp:0(2'beta1_power/Initializer/initial_value:0
q
beta2_power:0beta2_power/Assign!beta2_power/Read/ReadVariableOp:0(2'beta2_power/Initializer/initial_value:0
u
intercept/Adam:0intercept/Adam/Assign$intercept/Adam/Read/ReadVariableOp:0(2"intercept/Adam/Initializer/zeros:0
}
intercept/Adam_1:0intercept/Adam_1/Assign&intercept/Adam_1/Read/ReadVariableOp:0(2$intercept/Adam_1/Initializer/zeros:0
e
slope/Adam:0slope/Adam/Assign slope/Adam/Read/ReadVariableOp:0(2slope/Adam/Initializer/zeros:0
m
slope/Adam_1:0slope/Adam_1/Assign"slope/Adam_1/Read/ReadVariableOp:0(2 slope/Adam_1/Initializer/zeros:0
}
intercept_1/Adam:0intercept_1/Adam/Assign&intercept_1/Adam/Read/ReadVariableOp:0(2$intercept_1/Adam/Initializer/zeros:0
Е
intercept_1/Adam_1:0intercept_1/Adam_1/Assign(intercept_1/Adam_1/Read/ReadVariableOp:0(2&intercept_1/Adam_1/Initializer/zeros:0
m
slope_1/Adam:0slope_1/Adam/Assign"slope_1/Adam/Read/ReadVariableOp:0(2 slope_1/Adam/Initializer/zeros:0
u
slope_1/Adam_1:0slope_1/Adam_1/Assign$slope_1/Adam_1/Read/ReadVariableOp:0(2"slope_1/Adam_1/Initializer/zeros:0
}
intercept_2/Adam:0intercept_2/Adam/Assign&intercept_2/Adam/Read/ReadVariableOp:0(2$intercept_2/Adam/Initializer/zeros:0
Е
intercept_2/Adam_1:0intercept_2/Adam_1/Assign(intercept_2/Adam_1/Read/ReadVariableOp:0(2&intercept_2/Adam_1/Initializer/zeros:0
m
slope_2/Adam:0slope_2/Adam/Assign"slope_2/Adam/Read/ReadVariableOp:0(2 slope_2/Adam/Initializer/zeros:0
u
slope_2/Adam_1:0slope_2/Adam_1/Assign$slope_2/Adam_1/Read/ReadVariableOp:0(2"slope_2/Adam_1/Initializer/zeros:0
}
intercept_3/Adam:0intercept_3/Adam/Assign&intercept_3/Adam/Read/ReadVariableOp:0(2$intercept_3/Adam/Initializer/zeros:0
Е
intercept_3/Adam_1:0intercept_3/Adam_1/Assign(intercept_3/Adam_1/Read/ReadVariableOp:0(2&intercept_3/Adam_1/Initializer/zeros:0
m
slope_3/Adam:0slope_3/Adam/Assign"slope_3/Adam/Read/ReadVariableOp:0(2 slope_3/Adam/Initializer/zeros:0
u
slope_3/Adam_1:0slope_3/Adam_1/Assign$slope_3/Adam_1/Read/ReadVariableOp:0(2"slope_3/Adam_1/Initializer/zeros:0
}
intercept_4/Adam:0intercept_4/Adam/Assign&intercept_4/Adam/Read/ReadVariableOp:0(2$intercept_4/Adam/Initializer/zeros:0
Е
intercept_4/Adam_1:0intercept_4/Adam_1/Assign(intercept_4/Adam_1/Read/ReadVariableOp:0(2&intercept_4/Adam_1/Initializer/zeros:0
m
slope_4/Adam:0slope_4/Adam/Assign"slope_4/Adam/Read/ReadVariableOp:0(2 slope_4/Adam/Initializer/zeros:0
u
slope_4/Adam_1:0slope_4/Adam_1/Assign$slope_4/Adam_1/Read/ReadVariableOp:0(2"slope_4/Adam_1/Initializer/zeros:0
}
intercept_5/Adam:0intercept_5/Adam/Assign&intercept_5/Adam/Read/ReadVariableOp:0(2$intercept_5/Adam/Initializer/zeros:0
Е
intercept_5/Adam_1:0intercept_5/Adam_1/Assign(intercept_5/Adam_1/Read/ReadVariableOp:0(2&intercept_5/Adam_1/Initializer/zeros:0
m
slope_5/Adam:0slope_5/Adam/Assign"slope_5/Adam/Read/ReadVariableOp:0(2 slope_5/Adam/Initializer/zeros:0
u
slope_5/Adam_1:0slope_5/Adam_1/Assign$slope_5/Adam_1/Read/ReadVariableOp:0(2"slope_5/Adam_1/Initializer/zeros:0
Е
conv2d/kernel/Adam:0conv2d/kernel/Adam/Assign(conv2d/kernel/Adam/Read/ReadVariableOp:0(2&conv2d/kernel/Adam/Initializer/zeros:0
Н
conv2d/kernel/Adam_1:0conv2d/kernel/Adam_1/Assign*conv2d/kernel/Adam_1/Read/ReadVariableOp:0(2(conv2d/kernel/Adam_1/Initializer/zeros:0
Н
conv2d_1/kernel/Adam:0conv2d_1/kernel/Adam/Assign*conv2d_1/kernel/Adam/Read/ReadVariableOp:0(2(conv2d_1/kernel/Adam/Initializer/zeros:0
Х
conv2d_1/kernel/Adam_1:0conv2d_1/kernel/Adam_1/Assign,conv2d_1/kernel/Adam_1/Read/ReadVariableOp:0(2*conv2d_1/kernel/Adam_1/Initializer/zeros:0
Н
conv2d_2/kernel/Adam:0conv2d_2/kernel/Adam/Assign*conv2d_2/kernel/Adam/Read/ReadVariableOp:0(2(conv2d_2/kernel/Adam/Initializer/zeros:0
Х
conv2d_2/kernel/Adam_1:0conv2d_2/kernel/Adam_1/Assign,conv2d_2/kernel/Adam_1/Read/ReadVariableOp:0(2*conv2d_2/kernel/Adam_1/Initializer/zeros:0
Н
conv2d_3/kernel/Adam:0conv2d_3/kernel/Adam/Assign*conv2d_3/kernel/Adam/Read/ReadVariableOp:0(2(conv2d_3/kernel/Adam/Initializer/zeros:0
Х
conv2d_3/kernel/Adam_1:0conv2d_3/kernel/Adam_1/Assign,conv2d_3/kernel/Adam_1/Read/ReadVariableOp:0(2*conv2d_3/kernel/Adam_1/Initializer/zeros:0
Н
conv2d_4/kernel/Adam:0conv2d_4/kernel/Adam/Assign*conv2d_4/kernel/Adam/Read/ReadVariableOp:0(2(conv2d_4/kernel/Adam/Initializer/zeros:0
Х
conv2d_4/kernel/Adam_1:0conv2d_4/kernel/Adam_1/Assign,conv2d_4/kernel/Adam_1/Read/ReadVariableOp:0(2*conv2d_4/kernel/Adam_1/Initializer/zeros:0*щ
serving_default╒
=
Placeholder:0,
Placeholder:0         :
flatten_1_hp*
flatten_1/Reshape:0         
<
flatten_1_lp,
flatten_1/Reshape_1:0         
tensorflow/serving/predict
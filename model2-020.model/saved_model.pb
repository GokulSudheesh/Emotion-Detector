м
Ћ§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8ѓ	
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
shape: *
dtype0*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
: 
n
conv2d/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d/bias*
shape: 
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
: 

conv2d_1/kernelVarHandleOp*
shape:  *
dtype0*
_output_shapes
: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:  
r
conv2d_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_1/bias*
shape: 
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
: 

conv2d_2/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_nameconv2d_2/kernel*
shape: @
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*&
_output_shapes
: @
r
conv2d_2/biasVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
shape:@@*
dtype0* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*
dtype0*&
_output_shapes
:@@
r
conv2d_3/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
dtype0*
_output_shapes
:@
v
dense/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
H*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0* 
_output_shapes
:
H
m

dense/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes	
:
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
shape:	*
dtype0*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	*
dtype0
p
dense_1/biasVarHandleOp*
dtype0*
shape:*
_output_shapes
: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
f
	Adam/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
shape: *
dtype0*
_output_shapes
: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/conv2d/kernel/mVarHandleOp*
shape: *
dtype0*
_output_shapes
: *%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
shape:  *'
shared_nameAdam/conv2d_1/kernel/m*
dtype0

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:  *
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
shape: *%
shared_nameAdam/conv2d_1/bias/m*
dtype0
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
shape: @*'
shared_nameAdam/conv2d_2/kernel/m*
dtype0

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv2d_2/bias/mVarHandleOp*
shape:@*
_output_shapes
: *%
shared_nameAdam/conv2d_2/bias/m*
dtype0
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_3/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *'
shared_nameAdam/conv2d_3/kernel/m*
shape:@@

*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*
dtype0*&
_output_shapes
:@@

Adam/conv2d_3/bias/mVarHandleOp*
dtype0*
_output_shapes
: *%
shared_nameAdam/conv2d_3/bias/m*
shape:@
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
dtype0*
_output_shapes
:@

Adam/dense/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *$
shared_nameAdam/dense/kernel/m*
shape:
H
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
H*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *"
shared_nameAdam/dense/bias/m*
shape:*
dtype0
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*&
shared_nameAdam/dense_1/kernel/m*
shape:	

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *$
shared_nameAdam/dense_1/bias/m*
shape:*
dtype0
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *%
shared_nameAdam/conv2d/kernel/v*
shape: *
dtype0

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *#
shared_nameAdam/conv2d/bias/v*
shape: *
dtype0
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *'
shared_nameAdam/conv2d_1/kernel/v*
shape:  *
dtype0

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:  *
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *%
shared_nameAdam/conv2d_1/bias/v*
shape: *
dtype0
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
dtype0*
_output_shapes
: 

Adam/conv2d_2/kernel/vVarHandleOp*'
shared_nameAdam/conv2d_2/kernel/v*
shape: @*
dtype0*
_output_shapes
: 

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv2d_2/bias/vVarHandleOp*%
shared_nameAdam/conv2d_2/bias/v*
shape:@*
_output_shapes
: *
dtype0
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_3/kernel/vVarHandleOp*'
shared_nameAdam/conv2d_3/kernel/v*
shape:@@*
_output_shapes
: *
dtype0

*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_3/bias/vVarHandleOp*%
shared_nameAdam/conv2d_3/bias/v*
shape:@*
_output_shapes
: *
dtype0
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
dtype0*
_output_shapes
:@

Adam/dense/kernel/vVarHandleOp*$
shared_nameAdam/dense/kernel/v*
shape:
H*
dtype0*
_output_shapes
: 
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
dtype0* 
_output_shapes
:
H
{
Adam/dense/bias/vVarHandleOp*"
shared_nameAdam/dense/bias/v*
shape:*
dtype0*
_output_shapes
: 
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
dtype0*
_output_shapes	
:

Adam/dense_1/kernel/vVarHandleOp*&
shared_nameAdam/dense_1/kernel/v*
shape:	*
dtype0*
_output_shapes
: 

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
dtype0*
_output_shapes
:	
~
Adam/dense_1/bias/vVarHandleOp*$
shared_nameAdam/dense_1/bias/v*
shape:*
dtype0*
_output_shapes
: 
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
ЄM
ConstConst"/device:CPU:0*пL
valueеLBвL BЫL
Й
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
R
$trainable_variables
%	variables
&regularization_losses
'	keras_api
R
(trainable_variables
)	variables
*regularization_losses
+	keras_api
h

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
h

2kernel
3bias
4trainable_variables
5	variables
6regularization_losses
7	keras_api
R
8trainable_variables
9	variables
:regularization_losses
;	keras_api
R
<trainable_variables
=	variables
>regularization_losses
?	keras_api
R
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
h

Dkernel
Ebias
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
R
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
h

Nkernel
Obias
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
А
Titer

Ubeta_1

Vbeta_2
	Wdecay
Xlearning_ratemmmm ,mЁ-mЂ2mЃ3mЄDmЅEmІNmЇOmЈvЉvЊvЋvЌ,v­-vЎ2vЏ3vАDvБEvВNvГOvД
V
0
1
2
3
,4
-5
26
37
D8
E9
N10
O11
V
0
1
2
3
,4
-5
26
37
D8
E9
N10
O11
 

trainable_variables
Ymetrics

Zlayers
	variables
regularization_losses
[layer_regularization_losses
\non_trainable_variables
 
 
 
 

trainable_variables
]metrics

^layers
	variables
regularization_losses
_layer_regularization_losses
`non_trainable_variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 

trainable_variables
ametrics

blayers
	variables
regularization_losses
clayer_regularization_losses
dnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 

 trainable_variables
emetrics

flayers
!	variables
"regularization_losses
glayer_regularization_losses
hnon_trainable_variables
 
 
 

$trainable_variables
imetrics

jlayers
%	variables
&regularization_losses
klayer_regularization_losses
lnon_trainable_variables
 
 
 

(trainable_variables
mmetrics

nlayers
)	variables
*regularization_losses
olayer_regularization_losses
pnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 

.trainable_variables
qmetrics

rlayers
/	variables
0regularization_losses
slayer_regularization_losses
tnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31

20
31
 

4trainable_variables
umetrics

vlayers
5	variables
6regularization_losses
wlayer_regularization_losses
xnon_trainable_variables
 
 
 

8trainable_variables
ymetrics

zlayers
9	variables
:regularization_losses
{layer_regularization_losses
|non_trainable_variables
 
 
 

<trainable_variables
}metrics

~layers
=	variables
>regularization_losses
layer_regularization_losses
non_trainable_variables
 
 
 

@trainable_variables
metrics
layers
A	variables
Bregularization_losses
 layer_regularization_losses
non_trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1

D0
E1
 

Ftrainable_variables
metrics
layers
G	variables
Hregularization_losses
 layer_regularization_losses
non_trainable_variables
 
 
 

Jtrainable_variables
metrics
layers
K	variables
Lregularization_losses
 layer_regularization_losses
non_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

N0
O1

N0
O1
 

Ptrainable_variables
metrics
layers
Q	variables
Rregularization_losses
 layer_regularization_losses
non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
V
0
1
2
3
4
5
6
	7

8
9
10
11
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


total

count

_fn_kwargs
trainable_variables
	variables
regularization_losses
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
Ё
trainable_variables
metrics
layers
	variables
regularization_losses
 layer_regularization_losses
non_trainable_variables
 
 
 

0
1
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 

serving_default_conv2d_inputPlaceholder*$
shape:џџџџџџџџџ00*
dtype0*/
_output_shapes
:џџџџџџџџџ00
о
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*,
_gradient_op_typePartitionedCall-20382*,
f'R%
#__inference_signature_wrapper_19953**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:џџџџџџџџџ*
Tout
2*
Tin
2
O
saver_filenamePlaceholder*
shape: *
_output_shapes
: *
dtype0
Ђ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*,
_gradient_op_typePartitionedCall-20447*'
f"R 
__inference__traced_save_20446**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *
Tout
2*8
Tin1
/2-	
С
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*,
_gradient_op_typePartitionedCall-20589**
f%R#
!__inference__traced_restore_20588**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *
Tout
2*7
Tin0
.2,ЭЕ
Ь
E
)__inference_dropout_1_layer_call_fn_20210

inputs
identityЁ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-19680**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tout
2*
Tin
2*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_19668h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs

`
B__inference_dropout_layer_call_and_return_conditional_losses_20165

inputs

identity_1V
IdentityIdentityinputs*/
_output_shapes
:џџџџџџџџџ *
T0c

Identity_1IdentityIdentity:output:0*/
_output_shapes
:џџџџџџџџџ *
T0"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
Б
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_19752

inputs
identityQ
dropout/rateConst*
_output_shapes
: *
valueB
 *   ?*
dtype0C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0_
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*(
_output_shapes
:џџџџџџџџџ*
T0*
dtype0
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Ѓ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*(
_output_shapes
:џџџџџџџџџ*
T0
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*(
_output_shapes
:џџџџџџџџџ*
T0R
dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*(
_output_shapes
:џџџџџџџџџ*
T0b
dropout/mulMulinputsdropout/truediv:z:0*(
_output_shapes
:џџџџџџџџџ*
T0p
dropout/CastCastdropout/GreaterEqual:z:0*(
_output_shapes
:џџџџџџџџџ*

SrcT0
*

DstT0j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*(
_output_shapes
:џџџџџџџџџ*
T0Z
IdentityIdentitydropout/mul_1:z:0*(
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
№
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_20195

inputs
identityQ
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  >C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0_
dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*/
_output_shapes
:џџџџџџџџџ@*
T0*
dtype0
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Њ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*/
_output_shapes
:џџџџџџџџџ@*
T0
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*/
_output_shapes
:џџџџџџџџџ@*
T0R
dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*/
_output_shapes
:џџџџџџџџџ@*
T0i
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:џџџџџџџџџ@*
T0w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ@q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
Ѓ
Љ
(__inference_conv2d_3_layer_call_fn_19560

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_19549*,
_gradient_op_typePartitionedCall-19555*
Tin
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
е	
л
B__inference_dense_1_layer_call_and_return_conditional_losses_20285

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЃ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
ЙЃ

!__inference__traced_restore_20588
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias&
"assignvariableop_4_conv2d_2_kernel$
 assignvariableop_5_conv2d_2_bias&
"assignvariableop_6_conv2d_3_kernel$
 assignvariableop_7_conv2d_3_bias#
assignvariableop_8_dense_kernel!
assignvariableop_9_dense_bias&
"assignvariableop_10_dense_1_kernel$
 assignvariableop_11_dense_1_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count,
(assignvariableop_19_adam_conv2d_kernel_m*
&assignvariableop_20_adam_conv2d_bias_m.
*assignvariableop_21_adam_conv2d_1_kernel_m,
(assignvariableop_22_adam_conv2d_1_bias_m.
*assignvariableop_23_adam_conv2d_2_kernel_m,
(assignvariableop_24_adam_conv2d_2_bias_m.
*assignvariableop_25_adam_conv2d_3_kernel_m,
(assignvariableop_26_adam_conv2d_3_bias_m+
'assignvariableop_27_adam_dense_kernel_m)
%assignvariableop_28_adam_dense_bias_m-
)assignvariableop_29_adam_dense_1_kernel_m+
'assignvariableop_30_adam_dense_1_bias_m,
(assignvariableop_31_adam_conv2d_kernel_v*
&assignvariableop_32_adam_conv2d_bias_v.
*assignvariableop_33_adam_conv2d_1_kernel_v,
(assignvariableop_34_adam_conv2d_1_bias_v.
*assignvariableop_35_adam_conv2d_2_kernel_v,
(assignvariableop_36_adam_conv2d_2_bias_v.
*assignvariableop_37_adam_conv2d_3_kernel_v,
(assignvariableop_38_adam_conv2d_3_bias_v+
'assignvariableop_39_adam_dense_kernel_v)
%assignvariableop_40_adam_dense_bias_v-
)assignvariableop_41_adam_dense_1_kernel_v+
'assignvariableop_42_adam_dense_1_bias_v
identity_44ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1
RestoreV2/tensor_namesConst"/device:CPU:0*Т
valueИBЕ+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:+Ц
RestoreV2/shape_and_slicesConst"/device:CPU:0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+ј
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*9
dtypes/
-2+	*Т
_output_shapesЏ
Ќ:::::::::::::::::::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:z
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:~
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:}
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0*
dtype0	*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:{
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:{
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_conv2d_kernel_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_conv2d_bias_mIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_1_kernel_mIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_1_bias_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
_output_shapes
:*
T0
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_2_kernel_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
_output_shapes
:*
T0
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_2_bias_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
_output_shapes
:*
T0
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_3_kernel_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
_output_shapes
:*
T0
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_3_bias_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
_output_shapes
:*
T0
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_kernel_mIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
_output_shapes
:*
T0
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_dense_bias_mIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_mIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_mIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_conv2d_kernel_vIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_conv2d_bias_vIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_1_kernel_vIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_1_bias_vIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_2_kernel_vIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv2d_2_bias_vIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_3_kernel_vIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_3_bias_vIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_dense_kernel_vIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_dense_bias_vIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_1_kernel_vIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_1_bias_vIdentity_42:output:0*
dtype0*
_output_shapes
 
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:Е
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 
Identity_44IdentityIdentity_43:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_44Identity_44:output:0*У
_input_shapesБ
Ў: :::::::::::::::::::::::::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_27AssignVariableOp_272$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2:$ : : :  : : : : :( : :
 : :' : : :# : : : :	 :+ : : : :& : :+ '
%
_user_specified_namefile_prefix:" : : : : :* :% : : :! : : : : :) : : 

d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_19501

inputs
identityЂ
MaxPoolMaxPoolinputs*
strides
*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
ksize
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
ЮQ
Э
__inference__traced_save_20446
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_51a26580c778471ca68b115a8aaab68b/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*Т
valueИBЕ+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0У
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0ь
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop"/device:CPU:0*9
dtypes/
-2+	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
value	B :*
dtype0
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:У
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 Й
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
_output_shapes
:*
T0*
N
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*Г
_input_shapesЁ
: : : :  : : @:@:@@:@:
H::	:: : : : : : : : : :  : : @:@:@@:@:
H::	:: : :  : : @:@:@@:@:
H::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:$ : : :  : : :, : : :( : :
 : :' : : :# : : : :	 :+ : : : :& : :+ '
%
_user_specified_namefile_prefix:" : : : : :* :% : : :! : : : : :) : : 
е	
л
B__inference_dense_1_layer_call_and_return_conditional_losses_19787

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЃ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ*
T0 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ*
T0V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 

f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_19568

inputs
identityЂ
MaxPoolMaxPoolinputs*
paddingVALID*
strides
*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
ю
a
B__inference_dropout_layer_call_and_return_conditional_losses_19610

inputs
identityQ
dropout/rateConst*
valueB
 *  >*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:џџџџџџџџџ 
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Њ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ 
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:џџџџџџџџџ R
dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:џџџџџџџџџ i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*/
_output_shapes
:џџџџџџџџџ *

SrcT0
q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs


*__inference_sequential_layer_call_fn_19930
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityЂStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*,
_gradient_op_typePartitionedCall-19915**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:џџџџџџџџџ*
Tin
2*
Tout
2*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_19914
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :, (
&
_user_specified_nameconv2d_input: : : :
 : : : : :	 : 

b
D__inference_dropout_1_layer_call_and_return_conditional_losses_20200

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
Њ
K
/__inference_max_pooling2d_1_layer_call_fn_19577

inputs
identityТ
PartitionedCallPartitionedCallinputs*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_19568**
config_proto

GPU 

CPU2J 8*
Tout
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*,
_gradient_op_typePartitionedCall-19574*
Tin
2
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs


*__inference_sequential_layer_call_fn_19882
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityЂStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_19866**
config_proto

GPU 

CPU2J 8*
Tout
2*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-19867*
Tin
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :, (
&
_user_specified_nameconv2d_input: : : :
 : : : : :	 : 
О1
Ц
E__inference_sequential_layer_call_and_return_conditional_losses_19835
conv2d_input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_19457*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-19463*/
_output_shapes
:џџџџџџџџџ00 Ў
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_19482*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-19488*/
_output_shapes
:џџџџџџџџџ00 ж
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-19507*/
_output_shapes
:џџџџџџџџџ *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_19501**
config_proto

GPU 

CPU2J 8Ч
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-19629*/
_output_shapes
:џџџџџџџџџ *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_19617**
config_proto

GPU 

CPU2J 8Ї
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-19530*/
_output_shapes
:џџџџџџџџџ@*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_19524**
config_proto

GPU 

CPU2J 8А
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-19555*/
_output_shapes
:џџџџџџџџџ@*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_19549**
config_proto

GPU 

CPU2J 8к
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*/
_output_shapes
:џџџџџџџџџ@*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-19574**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_19568Э
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_19668*
Tout
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-19680*
Tin
2*/
_output_shapes
:џџџџџџџџџ@М
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_19691*
Tout
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-19697*
Tin
2*(
_output_shapes
:џџџџџџџџџH
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_19715*
Tout
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-19721*
Tin
2*(
_output_shapes
:џџџџџџџџџФ
dropout_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*(
_output_shapes
:џџџџџџџџџ*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-19771**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_19759
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*'
_output_shapes
:џџџџџџџџџ*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-19793**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_19787М
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00::::::::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall: : :, (
&
_user_specified_nameconv2d_input: : : :
 : : : : :	 : 
љ
^
B__inference_flatten_layer_call_and_return_conditional_losses_19691

inputs
identity^
Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ $  e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:џџџџџџџџџH*
T0Y
IdentityIdentityReshape:output:0*(
_output_shapes
:џџџџџџџџџH*
T0"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs

м
C__inference_conv2d_1_layer_call_and_return_conditional_losses_19482

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЊ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: 
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Ѕ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ "
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 

b
D__inference_dropout_2_layer_call_and_return_conditional_losses_19759

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:џџџџџџџџџ*
T0\

Identity_1IdentityIdentity:output:0*(
_output_shapes
:џџџџџџџџџ*
T0"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
Џ@

E__inference_sequential_layer_call_and_return_conditional_losses_20106

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpЂconv2d_3/BiasAdd/ReadVariableOpЂconv2d_3/Conv2D/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpИ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: *
dtype0Ї
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ00 *
paddingSAME*
T0*
strides
Ў
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ00 f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ00 М
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  О
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ00 *
paddingSAME*
strides
В
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ00 j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ00 *
T0Њ
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
strides
*
paddingVALID*
ksize
v
dropout/IdentityIdentitymax_pooling2d/MaxPool:output:0*/
_output_shapes
:џџџџџџџџџ *
T0М
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: @*
dtype0О
conv2d_2/Conv2DConv2Ddropout/Identity:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ@*
strides
*
paddingSAME*
T0В
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@М
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@@Р
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
strides
*
paddingSAMEВ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Ќ
max_pooling2d_1/MaxPoolMaxPoolconv2d_3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
strides
*
ksize
*
paddingVALIDz
dropout_1/IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@f
flatten/Reshape/shapeConst*
valueB"џџџџ $  *
dtype0*
_output_shapes
:
flatten/ReshapeReshapedropout_1/Identity:output:0flatten/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџHА
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
H
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ­
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџk
dropout_2/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџГ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_1/MatMulMatMuldropout_2/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџА
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџf
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџч
IdentityIdentitydense_1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00::::::::::::2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs: : : :
 : : : : :	 : 
К
C
'__inference_flatten_layer_call_fn_20221

inputs
identity
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*,
_gradient_op_typePartitionedCall-19697*(
_output_shapes
:џџџџџџџџџH*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_19691a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџH"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
Ь
`
'__inference_dropout_layer_call_fn_20170

inputs
identityЂStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputs*,
_gradient_op_typePartitionedCall-19621**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:џџџџџџџџџ *
Tout
2*
Tin
2*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_19610
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ 22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ќ1
Р
E__inference_sequential_layer_call_and_return_conditional_losses_19914

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-19463*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_19457*/
_output_shapes
:џџџџџџџџџ00 *
Tin
2*
Tout
2Ў
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-19488**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_19482*/
_output_shapes
:џџџџџџџџџ00 *
Tin
2*
Tout
2ж
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-19507**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:џџџџџџџџџ *
Tout
2*
Tin
2*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_19501Ч
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-19629**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:џџџџџџџџџ *
Tout
2*
Tin
2*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_19617Ї
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-19530**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tout
2*
Tin
2*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_19524А
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-19555**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tout
2*
Tin
2*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_19549к
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tout
2*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_19568*
Tin
2*,
_gradient_op_typePartitionedCall-19574*/
_output_shapes
:џџџџџџџџџ@**
config_proto

GPU 

CPU2J 8Э
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tout
2*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_19668*
Tin
2*,
_gradient_op_typePartitionedCall-19680*/
_output_shapes
:џџџџџџџџџ@**
config_proto

GPU 

CPU2J 8М
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tout
2*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_19691*
Tin
2*,
_gradient_op_typePartitionedCall-19697*(
_output_shapes
:џџџџџџџџџH**
config_proto

GPU 

CPU2J 8
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tout
2*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_19715*
Tin
2*,
_gradient_op_typePartitionedCall-19721*(
_output_shapes
:џџџџџџџџџ**
config_proto

GPU 

CPU2J 8Ф
dropout_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџ*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_19759*,
_gradient_op_typePartitionedCall-19771*
Tout
2
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:џџџџџџџџџ*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_19787*,
_gradient_op_typePartitionedCall-19793*
Tout
2М
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00::::::::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall: : : : :	 : : : :& "
 
_user_specified_nameinputs: : : :
 

м
C__inference_conv2d_3_layer_call_and_return_conditional_losses_19549

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЊ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:@@*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T0 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Ѕ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 

b
D__inference_dropout_1_layer_call_and_return_conditional_losses_19668

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
д	
й
@__inference_dense_layer_call_and_return_conditional_losses_19715

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЄ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
H*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџH::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
у

#__inference_signature_wrapper_19953
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*)
f$R"
 __inference__wrapped_model_19443*
Tout
2*,
_gradient_op_typePartitionedCall-19938*
Tin
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:џџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :	 : : : :, (
&
_user_specified_nameconv2d_input: : : :
 
шL
щ	
 __inference__wrapped_model_19443
conv2d_input4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource6
2sequential_conv2d_3_conv2d_readvariableop_resource7
3sequential_conv2d_3_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identityЂ(sequential/conv2d/BiasAdd/ReadVariableOpЂ'sequential/conv2d/Conv2D/ReadVariableOpЂ*sequential/conv2d_1/BiasAdd/ReadVariableOpЂ)sequential/conv2d_1/Conv2D/ReadVariableOpЂ*sequential/conv2d_2/BiasAdd/ReadVariableOpЂ)sequential/conv2d_2/Conv2D/ReadVariableOpЂ*sequential/conv2d_3/BiasAdd/ReadVariableOpЂ)sequential/conv2d_3/Conv2D/ReadVariableOpЂ'sequential/dense/BiasAdd/ReadVariableOpЂ&sequential/dense/MatMul/ReadVariableOpЂ)sequential/dense_1/BiasAdd/ReadVariableOpЂ(sequential/dense_1/MatMul/ReadVariableOpЮ
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: *
dtype0У
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ00 *
paddingSAME*
strides
Ф
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0Г
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ00 *
T0|
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ00 *
T0в
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:  *
dtype0п
sequential/conv2d_1/Conv2DConv2D$sequential/conv2d/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ00 *
paddingSAME*
strides
Ш
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0Й
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ00 *
T0
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ00 *
T0Р
 sequential/max_pooling2d/MaxPoolMaxPool&sequential/conv2d_1/Relu:activations:0*
ksize
*/
_output_shapes
:џџџџџџџџџ *
strides
*
paddingVALID
sequential/dropout/IdentityIdentity)sequential/max_pooling2d/MaxPool:output:0*/
_output_shapes
:џџџџџџџџџ *
T0в
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: @*
dtype0п
sequential/conv2d_2/Conv2DConv2D$sequential/dropout/Identity:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
strides
*
paddingSAMEШ
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0Й
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ@*
T0
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ@*
T0в
)sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:@@*
dtype0с
sequential/conv2d_3/Conv2DConv2D&sequential/conv2d_2/Relu:activations:01sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
T0*
strides
Ш
*sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0Й
sequential/conv2d_3/BiasAddBiasAdd#sequential/conv2d_3/Conv2D:output:02sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
sequential/conv2d_3/ReluRelu$sequential/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Т
"sequential/max_pooling2d_1/MaxPoolMaxPool&sequential/conv2d_3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
*
ksize

sequential/dropout_1/IdentityIdentity+sequential/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@q
 sequential/flatten/Reshape/shapeConst*
valueB"џџџџ $  *
dtype0*
_output_shapes
:Ћ
sequential/flatten/ReshapeReshape&sequential/dropout_1/Identity:output:0)sequential/flatten/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџHЦ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
HЉ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџУ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Њ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
sequential/dropout_2/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџЩ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Џ
sequential/dense_1/MatMulMatMul&sequential/dropout_2/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЦ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Џ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџі
IdentityIdentity$sequential/dense_1/Softmax:softmax:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp+^sequential/conv2d_3/BiasAdd/ReadVariableOp*^sequential/conv2d_3/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00::::::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2X
*sequential/conv2d_3/BiasAdd/ReadVariableOp*sequential/conv2d_3/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/conv2d_3/Conv2D/ReadVariableOp)sequential/conv2d_3/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp: : : : :	 : : : :, (
&
_user_specified_nameconv2d_input: : : :
 
є5
А
E__inference_sequential_layer_call_and_return_conditional_losses_19805
conv2d_input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2*,
_gradient_op_typePartitionedCall-19463*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_19457*
Tin
2**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:џџџџџџџџџ00 Ў
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tout
2*,
_gradient_op_typePartitionedCall-19488*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_19482*
Tin
2**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:џџџџџџџџџ00 ж
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-19507*
Tin
2*/
_output_shapes
:џџџџџџџџџ *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_19501*
Tout
2**
config_proto

GPU 

CPU2J 8з
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tout
2*,
_gradient_op_typePartitionedCall-19621**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_19610*/
_output_shapes
:џџџџџџџџџ *
Tin
2Џ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-19530*
Tin
2*/
_output_shapes
:џџџџџџџџџ@*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_19524*
Tout
2**
config_proto

GPU 

CPU2J 8А
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-19555*
Tin
2*/
_output_shapes
:џџџџџџџџџ@*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_19549*
Tout
2**
config_proto

GPU 

CPU2J 8к
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tout
2*,
_gradient_op_typePartitionedCall-19574**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_19568*/
_output_shapes
:џџџџџџџџџ@*
Tin
2џ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tout
2*,
_gradient_op_typePartitionedCall-19672**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_19661*/
_output_shapes
:џџџџџџџџџ@*
Tin
2Ф
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-19697*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_19691*(
_output_shapes
:џџџџџџџџџH*
Tin
2
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-19721*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_19715*(
_output_shapes
:џџџџџџџџџ*
Tin
2ј
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tout
2*
Tin
2*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_19752*,
_gradient_op_typePartitionedCall-19763**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџЅ
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_19787*,
_gradient_op_typePartitionedCall-19793**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:џџџџџџџџџІ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00::::::::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall: : : : :	 : : : :, (
&
_user_specified_nameconv2d_input: : : :
 
§

*__inference_sequential_layer_call_fn_20140

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityЂStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*'
_output_shapes
:џџџџџџџџџ**
config_proto

GPU 

CPU2J 8*
Tout
2*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_19914*
Tin
2*,
_gradient_op_typePartitionedCall-19915
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :	 : : : :& "
 
_user_specified_nameinputs: : : :
 
д	
й
@__inference_dense_layer_call_and_return_conditional_losses_20232

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЄ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
H*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:џџџџџџџџџ*
T0Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџH::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
Л
b
)__inference_dropout_2_layer_call_fn_20269

inputs
identityЂStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputs*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_19752*,
_gradient_op_typePartitionedCall-19763*
Tin
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
т5
Њ
E__inference_sequential_layer_call_and_return_conditional_losses_19866

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_19457*
Tout
2*/
_output_shapes
:џџџџџџџџџ00 *
Tin
2*,
_gradient_op_typePartitionedCall-19463Ў
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-19488*/
_output_shapes
:џџџџџџџџџ00 *
Tin
2*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_19482*
Tout
2**
config_proto

GPU 

CPU2J 8ж
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_19501*
Tout
2*/
_output_shapes
:џџџџџџџџџ *
Tin
2*,
_gradient_op_typePartitionedCall-19507з
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_19610*
Tout
2*/
_output_shapes
:џџџџџџџџџ *
Tin
2*,
_gradient_op_typePartitionedCall-19621Џ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-19530*/
_output_shapes
:џџџџџџџџџ@*
Tin
2*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_19524*
Tout
2**
config_proto

GPU 

CPU2J 8А
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-19555*/
_output_shapes
:џџџџџџџџџ@*
Tin
2*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_19549*
Tout
2**
config_proto

GPU 

CPU2J 8к
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:џџџџџџџџџ@*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_19568*
Tout
2*,
_gradient_op_typePartitionedCall-19574*
Tin
2џ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*,
_gradient_op_typePartitionedCall-19672*/
_output_shapes
:џџџџџџџџџ@*
Tin
2*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_19661*
Tout
2**
config_proto

GPU 

CPU2J 8Ф
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџH*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_19691*
Tout
2*,
_gradient_op_typePartitionedCall-19697*
Tin
2
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_19715*
Tout
2*,
_gradient_op_typePartitionedCall-19721*
Tin
2ј
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_19752*
Tout
2*,
_gradient_op_typePartitionedCall-19763*
Tin
2Ѕ
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-19793*
Tin
2**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_19787*
Tout
2І
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00::::::::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall: : : : :	 : : : :& "
 
_user_specified_nameinputs: : : :
 
ю
a
B__inference_dropout_layer_call_and_return_conditional_losses_20160

inputs
identityQ
dropout/rateConst*
_output_shapes
: *
valueB
 *  >*
dtype0C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0_
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*/
_output_shapes
:џџџџџџџџџ *
dtype0*
T0
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Њ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*/
_output_shapes
:џџџџџџџџџ *
T0
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*/
_output_shapes
:џџџџџџџџџ *
T0R
dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*/
_output_shapes
:џџџџџџџџџ *
T0i
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:џџџџџџџџџ *
T0w
dropout/CastCastdropout/GreaterEqual:z:0*/
_output_shapes
:џџџџџџџџџ *

SrcT0
*

DstT0q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*/
_output_shapes
:џџџџџџџџџ *
T0a
IdentityIdentitydropout/mul_1:z:0*/
_output_shapes
:џџџџџџџџџ *
T0"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
№
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_19661

inputs
identityQ
dropout/rateConst*
valueB
 *  >*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*
T0*/
_output_shapes
:џџџџџџџџџ@
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Њ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@R
dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:џџџџџџџџџ@q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
І
I
-__inference_max_pooling2d_layer_call_fn_19510

inputs
identityР
PartitionedCallPartitionedCallinputs*
Tout
2*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_19501*,
_gradient_op_typePartitionedCall-19507**
config_proto

GPU 

CPU2J 8*
Tin
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs

`
B__inference_dropout_layer_call_and_return_conditional_losses_19617

inputs

identity_1V
IdentityIdentityinputs*/
_output_shapes
:џџџџџџџџџ *
T0c

Identity_1IdentityIdentity:output:0*/
_output_shapes
:џџџџџџџџџ *
T0"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
Ѓ
Љ
(__inference_conv2d_1_layer_call_fn_19493

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_19482*,
_gradient_op_typePartitionedCall-19488*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
T0"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
Ѓ
Љ
(__inference_conv2d_2_layer_call_fn_19535

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_19524*,
_gradient_op_typePartitionedCall-19530*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
љ
^
B__inference_flatten_layer_call_and_return_conditional_losses_20216

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
valueB"џџџџ $  *
dtype0e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:џџџџџџџџџH*
T0Y
IdentityIdentityReshape:output:0*(
_output_shapes
:џџџџџџџџџH*
T0"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
тp

E__inference_sequential_layer_call_and_return_conditional_losses_20053

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpЂconv2d_3/BiasAdd/ReadVariableOpЂconv2d_3/Conv2D/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpИ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: *
dtype0Ї
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*
strides
*/
_output_shapes
:џџџџџџџџџ00 *
paddingSAMEЎ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ00 *
T0f
conv2d/ReluReluconv2d/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ00 *
T0М
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:  *
dtype0О
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*/
_output_shapes
:џџџџџџџџџ00 *
paddingSAMEВ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ00 j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ00 *
T0Њ
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*
ksize
*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
Y
dropout/dropout/rateConst*
_output_shapes
: *
valueB
 *  >*
dtype0c
dropout/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
_output_shapes
:*
T0g
"dropout/dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0g
"dropout/dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ?*
dtype0Є
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
dtype0*
T0*/
_output_shapes
:џџџџџџџџџ Є
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Т
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Д
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:џџџџџџџџџ Z
dropout/dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: ^
dropout/dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: Љ
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
dropout/dropout/mulMulmax_pooling2d/MaxPool:output:0dropout/dropout/truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ 
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:џџџџџџџџџ 
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ М
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @О
conv2d_2/Conv2DConv2Ddropout/dropout/mul_1:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*
T0*/
_output_shapes
:џџџџџџџџџ@В
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@М
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@@Р
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*
T0*/
_output_shapes
:џџџџџџџџџ@В
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Ќ
max_pooling2d_1/MaxPoolMaxPoolconv2d_3/Relu:activations:0*
ksize
*/
_output_shapes
:џџџџџџџџџ@*
strides
*
paddingVALID[
dropout_1/dropout/rateConst*
valueB
 *  >*
dtype0*
_output_shapes
: g
dropout_1/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:i
$dropout_1/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_1/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: Ј
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*/
_output_shapes
:џџџџџџџџџ@*
dtype0*
T0Њ
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Ш
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*/
_output_shapes
:џџџџџџџџџ@*
T0К
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*/
_output_shapes
:џџџџџџџџџ@*
T0\
dropout_1/dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
_output_shapes
: *
T0`
dropout_1/dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
_output_shapes
: *
T0Џ
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*/
_output_shapes
:џџџџџџџџџ@*
T0
dropout_1/dropout/mulMul max_pooling2d_1/MaxPool:output:0dropout_1/dropout/truediv:z:0*/
_output_shapes
:џџџџџџџџџ@*
T0
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ@
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@f
flatten/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ $  
flatten/ReshapeReshapedropout_1/dropout/mul_1:z:0flatten/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџHА
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
H
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ­
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ[
dropout_2/dropout/rateConst*
_output_shapes
: *
valueB
 *   ?*
dtype0_
dropout_2/dropout/ShapeShapedense/Relu:activations:0*
_output_shapes
:*
T0i
$dropout_2/dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0i
$dropout_2/dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ?*
dtype0Ё
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*(
_output_shapes
:џџџџџџџџџ*
dtype0*
T0Њ
$dropout_2/dropout/random_uniform/subSub-dropout_2/dropout/random_uniform/max:output:0-dropout_2/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0С
$dropout_2/dropout/random_uniform/mulMul7dropout_2/dropout/random_uniform/RandomUniform:output:0(dropout_2/dropout/random_uniform/sub:z:0*(
_output_shapes
:џџџџџџџџџ*
T0Г
 dropout_2/dropout/random_uniformAdd(dropout_2/dropout/random_uniform/mul:z:0-dropout_2/dropout/random_uniform/min:output:0*(
_output_shapes
:џџџџџџџџџ*
T0\
dropout_2/dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
dropout_2/dropout/subSub dropout_2/dropout/sub/x:output:0dropout_2/dropout/rate:output:0*
_output_shapes
: *
T0`
dropout_2/dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
dropout_2/dropout/truedivRealDiv$dropout_2/dropout/truediv/x:output:0dropout_2/dropout/sub:z:0*
T0*
_output_shapes
: Ј
dropout_2/dropout/GreaterEqualGreaterEqual$dropout_2/dropout/random_uniform:z:0dropout_2/dropout/rate:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dropout_2/dropout/mulMuldense/Relu:activations:0dropout_2/dropout/truediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:џџџџџџџџџ
dropout_2/dropout/mul_1Muldropout_2/dropout/mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџГ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_1/MatMulMatMuldropout_2/dropout/mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџА
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџf
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџч
IdentityIdentitydense_1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp: : : : :	 : : : :& "
 
_user_specified_nameinputs: : : :
 
Б
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_20259

inputs
identityQ
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *   ?C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0_
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*(
_output_shapes
:џџџџџџџџџ*
dtype0*
T0
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Ѓ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*(
_output_shapes
:џџџџџџџџџ*
T0
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*(
_output_shapes
:џџџџџџџџџ*
T0R
dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*(
_output_shapes
:џџџџџџџџџ*
T0b
dropout/mulMulinputsdropout/truediv:z:0*(
_output_shapes
:џџџџџџџџџ*
T0p
dropout/CastCastdropout/GreaterEqual:z:0*(
_output_shapes
:џџџџџџџџџ*

SrcT0
*

DstT0j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*(
_output_shapes
:џџџџџџџџџ*
T0Z
IdentityIdentitydropout/mul_1:z:0*(
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
в
І
%__inference_dense_layer_call_fn_20239

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*,
_gradient_op_typePartitionedCall-19721*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_19715**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџH::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 

Ї
&__inference_conv2d_layer_call_fn_19468

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_19457*,
_gradient_op_typePartitionedCall-19463*
Tin
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ "
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
З
E
)__inference_dropout_2_layer_call_fn_20274

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*(
_output_shapes
:џџџџџџџџџ*
Tout
2*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_19759**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-19771a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs

м
C__inference_conv2d_2_layer_call_and_return_conditional_losses_19524

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЊ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: @*
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T0 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T0j
ReluReluBiasAdd:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T0Ѕ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
Ш
C
'__inference_dropout_layer_call_fn_20175

inputs
identity
PartitionedCallPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:џџџџџџџџџ *
Tout
2*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_19617*,
_gradient_op_typePartitionedCall-19629*
Tin
2h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
§

*__inference_sequential_layer_call_fn_20123

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityЂStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:џџџџџџџџџ*
Tout
2*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_19866*,
_gradient_op_typePartitionedCall-19867*
Tin
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :	 : : : :& "
 
_user_specified_nameinputs: : : :
 

b
D__inference_dropout_2_layer_call_and_return_conditional_losses_20264

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:џџџџџџџџџ*
T0\

Identity_1IdentityIdentity:output:0*(
_output_shapes
:џџџџџџџџџ*
T0"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
д
Ј
'__inference_dense_1_layer_call_fn_20292

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*'
_output_shapes
:џџџџџџџџџ**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-19793*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_19787*
Tout
2*
Tin
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
а
b
)__inference_dropout_1_layer_call_fn_20205

inputs
identityЂStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputs*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_19661*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-19672*/
_output_shapes
:џџџџџџџџџ@
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:џџџџџџџџџ@*
T0"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

к
A__inference_conv2d_layer_call_and_return_conditional_losses_19457

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЊ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: *
dtype0Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
T0j
ReluReluBiasAdd:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
T0Ѕ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
T0"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*М
serving_defaultЈ
M
conv2d_input=
serving_default_conv2d_input:0џџџџџџџџџ00;
dense_10
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:дс
АJ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
Е_default_save_signature
Ж__call__
+З&call_and_return_all_conditional_losses"F
_tf_keras_sequentialћE{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 48, 48, 1], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 48, 48, 1], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Н
trainable_variables
	variables
regularization_losses
	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"Ќ
_tf_keras_layer{"class_name": "InputLayer", "name": "conv2d_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 48, 48, 1], "config": {"batch_input_shape": [null, 48, 48, 1], "dtype": "float32", "sparse": false, "name": "conv2d_input"}}


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
К__call__
+Л&call_and_return_all_conditional_losses"ї
_tf_keras_layerн{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 48, 48, 1], "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 48, 48, 1], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
ю

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
М__call__
+Н&call_and_return_all_conditional_losses"Ч
_tf_keras_layer­{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
ћ
$trainable_variables
%	variables
&regularization_losses
'	keras_api
О__call__
+П&call_and_return_all_conditional_losses"ъ
_tf_keras_layerа{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ў
(trainable_variables
)	variables
*regularization_losses
+	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
ю

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
Т__call__
+У&call_and_return_all_conditional_losses"Ч
_tf_keras_layer­{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
ю

2kernel
3bias
4trainable_variables
5	variables
6regularization_losses
7	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"Ч
_tf_keras_layer­{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
џ
8trainable_variables
9	variables
:regularization_losses
;	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses"ю
_tf_keras_layerд{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
В
<trainable_variables
=	variables
>regularization_losses
?	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses"Ё
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
Ў
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ђ

Dkernel
Ebias
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9216}}}}
Б
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
Ю__call__
+Я&call_and_return_all_conditional_losses" 
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
і

Nkernel
Obias
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
а__call__
+б&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
У
Titer

Ubeta_1

Vbeta_2
	Wdecay
Xlearning_ratemmmm ,mЁ-mЂ2mЃ3mЄDmЅEmІNmЇOmЈvЉvЊvЋvЌ,v­-vЎ2vЏ3vАDvБEvВNvГOvД"
	optimizer
v
0
1
2
3
,4
-5
26
37
D8
E9
N10
O11"
trackable_list_wrapper
v
0
1
2
3
,4
-5
26
37
D8
E9
N10
O11"
trackable_list_wrapper
 "
trackable_list_wrapper
Л
trainable_variables
Ymetrics

Zlayers
	variables
regularization_losses
[layer_regularization_losses
\non_trainable_variables
Ж__call__
Е_default_save_signature
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
-
вserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

trainable_variables
]metrics

^layers
	variables
regularization_losses
_layer_regularization_losses
`non_trainable_variables
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
':% 2conv2d/kernel
: 2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper

trainable_variables
ametrics

blayers
	variables
regularization_losses
clayer_regularization_losses
dnon_trainable_variables
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_1/kernel
: 2conv2d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper

 trainable_variables
emetrics

flayers
!	variables
"regularization_losses
glayer_regularization_losses
hnon_trainable_variables
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

$trainable_variables
imetrics

jlayers
%	variables
&regularization_losses
klayer_regularization_losses
lnon_trainable_variables
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

(trainable_variables
mmetrics

nlayers
)	variables
*regularization_losses
olayer_regularization_losses
pnon_trainable_variables
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_2/kernel
:@2conv2d_2/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper

.trainable_variables
qmetrics

rlayers
/	variables
0regularization_losses
slayer_regularization_losses
tnon_trainable_variables
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_3/kernel
:@2conv2d_3/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper

4trainable_variables
umetrics

vlayers
5	variables
6regularization_losses
wlayer_regularization_losses
xnon_trainable_variables
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

8trainable_variables
ymetrics

zlayers
9	variables
:regularization_losses
{layer_regularization_losses
|non_trainable_variables
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

<trainable_variables
}metrics

~layers
=	variables
>regularization_losses
layer_regularization_losses
non_trainable_variables
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
@trainable_variables
metrics
layers
A	variables
Bregularization_losses
 layer_regularization_losses
non_trainable_variables
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 :
H2dense/kernel
:2
dense/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
Ftrainable_variables
metrics
layers
G	variables
Hregularization_losses
 layer_regularization_losses
non_trainable_variables
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
Jtrainable_variables
metrics
layers
K	variables
Lregularization_losses
 layer_regularization_losses
non_trainable_variables
Ю__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_1/kernel
:2dense_1/bias
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
Ptrainable_variables
metrics
layers
Q	variables
Rregularization_losses
 layer_regularization_losses
non_trainable_variables
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(
0"
trackable_list_wrapper
v
0
1
2
3
4
5
6
	7

8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ѓ

total

count

_fn_kwargs
trainable_variables
	variables
regularization_losses
	keras_api
г__call__
+д&call_and_return_all_conditional_losses"х
_tf_keras_layerЫ{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
trainable_variables
metrics
layers
	variables
regularization_losses
 layer_regularization_losses
non_trainable_variables
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
,:* 2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
.:,  2Adam/conv2d_1/kernel/m
 : 2Adam/conv2d_1/bias/m
.:, @2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
.:,@@2Adam/conv2d_3/kernel/m
 :@2Adam/conv2d_3/bias/m
%:#
H2Adam/dense/kernel/m
:2Adam/dense/bias/m
&:$	2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
,:* 2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
.:,  2Adam/conv2d_1/kernel/v
 : 2Adam/conv2d_1/bias/v
.:, @2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
.:,@@2Adam/conv2d_3/kernel/v
 :@2Adam/conv2d_3/bias/v
%:#
H2Adam/dense/kernel/v
:2Adam/dense/bias/v
&:$	2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
ы2ш
 __inference__wrapped_model_19443У
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+
conv2d_inputџџџџџџџџџ00
і2ѓ
*__inference_sequential_layer_call_fn_19930
*__inference_sequential_layer_call_fn_20123
*__inference_sequential_layer_call_fn_19882
*__inference_sequential_layer_call_fn_20140Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
т2п
E__inference_sequential_layer_call_and_return_conditional_losses_19835
E__inference_sequential_layer_call_and_return_conditional_losses_19805
E__inference_sequential_layer_call_and_return_conditional_losses_20106
E__inference_sequential_layer_call_and_return_conditional_losses_20053Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
2
&__inference_conv2d_layer_call_fn_19468з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 2
A__inference_conv2d_layer_call_and_return_conditional_losses_19457з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
(__inference_conv2d_1_layer_call_fn_19493з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Ђ2
C__inference_conv2d_1_layer_call_and_return_conditional_losses_19482з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
-__inference_max_pooling2d_layer_call_fn_19510р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
А2­
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_19501р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
'__inference_dropout_layer_call_fn_20170
'__inference_dropout_layer_call_fn_20175Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Т2П
B__inference_dropout_layer_call_and_return_conditional_losses_20165
B__inference_dropout_layer_call_and_return_conditional_losses_20160Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
(__inference_conv2d_2_layer_call_fn_19535з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Ђ2
C__inference_conv2d_2_layer_call_and_return_conditional_losses_19524з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
(__inference_conv2d_3_layer_call_fn_19560з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Ђ2
C__inference_conv2d_3_layer_call_and_return_conditional_losses_19549з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
/__inference_max_pooling2d_1_layer_call_fn_19577р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
В2Џ
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_19568р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
)__inference_dropout_1_layer_call_fn_20205
)__inference_dropout_1_layer_call_fn_20210Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ц2У
D__inference_dropout_1_layer_call_and_return_conditional_losses_20200
D__inference_dropout_1_layer_call_and_return_conditional_losses_20195Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
б2Ю
'__inference_flatten_layer_call_fn_20221Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_flatten_layer_call_and_return_conditional_losses_20216Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Я2Ь
%__inference_dense_layer_call_fn_20239Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъ2ч
@__inference_dense_layer_call_and_return_conditional_losses_20232Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
)__inference_dropout_2_layer_call_fn_20269
)__inference_dropout_2_layer_call_fn_20274Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ц2У
D__inference_dropout_2_layer_call_and_return_conditional_losses_20259
D__inference_dropout_2_layer_call_and_return_conditional_losses_20264Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
б2Ю
'__inference_dense_1_layer_call_fn_20292Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_dense_1_layer_call_and_return_conditional_losses_20285Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
7B5
#__inference_signature_wrapper_19953conv2d_input
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 ~
)__inference_dropout_2_layer_call_fn_20274Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџА
(__inference_conv2d_1_layer_call_fn_19493IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
*__inference_sequential_layer_call_fn_20140i,-23DENO?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ00
p 

 
Њ "џџџџџџџџџЇ
B__inference_flatten_layer_call_and_return_conditional_losses_20216a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџH
 У
-__inference_max_pooling2d_layer_call_fn_19510RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџХ
E__inference_sequential_layer_call_and_return_conditional_losses_19805|,-23DENOEЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ00
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ў
&__inference_conv2d_layer_call_fn_19468IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Д
D__inference_dropout_1_layer_call_and_return_conditional_losses_20195l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ "-Ђ*
# 
0џџџџџџџџџ@
 А
(__inference_conv2d_2_layer_call_fn_19535,-IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@и
C__inference_conv2d_2_layer_call_and_return_conditional_losses_19524,-IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 И
#__inference_signature_wrapper_19953,-23DENOMЂJ
Ђ 
CЊ@
>
conv2d_input.+
conv2d_inputџџџџџџџџџ00"1Њ.
,
dense_1!
dense_1џџџџџџџџџЃ
B__inference_dense_1_layer_call_and_return_conditional_losses_20285]NO0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
*__inference_sequential_layer_call_fn_19930o,-23DENOEЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ00
p 

 
Њ "џџџџџџџџџЅ
 __inference__wrapped_model_19443,-23DENO=Ђ:
3Ђ0
.+
conv2d_inputџџџџџџџџџ00
Њ "1Њ.
,
dense_1!
dense_1џџџџџџџџџП
E__inference_sequential_layer_call_and_return_conditional_losses_20053v,-23DENO?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ00
p

 
Њ "%Ђ"

0џџџџџџџџџ
 и
C__inference_conv2d_1_layer_call_and_return_conditional_losses_19482IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 и
C__inference_conv2d_3_layer_call_and_return_conditional_losses_1954923IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
*__inference_sequential_layer_call_fn_20123i,-23DENO?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ00
p

 
Њ "џџџџџџџџџП
E__inference_sequential_layer_call_and_return_conditional_losses_20106v,-23DENO?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ00
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Х
/__inference_max_pooling2d_1_layer_call_fn_19577RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
'__inference_dropout_layer_call_fn_20170_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ " џџџџџџџџџ 
'__inference_dropout_layer_call_fn_20175_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ " џџџџџџџџџ 
)__inference_dropout_1_layer_call_fn_20205_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ " џџџџџџџџџ@І
D__inference_dropout_2_layer_call_and_return_conditional_losses_20264^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 ~
)__inference_dropout_2_layer_call_fn_20269Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџz
%__inference_dense_layer_call_fn_20239QDE0Ђ-
&Ђ#
!
inputsџџџџџџџџџH
Њ "џџџџџџџџџж
A__inference_conv2d_layer_call_and_return_conditional_losses_19457IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 І
D__inference_dropout_2_layer_call_and_return_conditional_losses_20259^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 В
B__inference_dropout_layer_call_and_return_conditional_losses_20160l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
)__inference_dropout_1_layer_call_fn_20210_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ " џџџџџџџџџ@Ђ
@__inference_dense_layer_call_and_return_conditional_losses_20232^DE0Ђ-
&Ђ#
!
inputsџџџџџџџџџH
Њ "&Ђ#

0џџџџџџџџџ
 Д
D__inference_dropout_1_layer_call_and_return_conditional_losses_20200l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ@
 В
B__inference_dropout_layer_call_and_return_conditional_losses_20165l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
*__inference_sequential_layer_call_fn_19882o,-23DENOEЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ00
p

 
Њ "џџџџџџџџџ{
'__inference_dense_1_layer_call_fn_20292PNO0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџы
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_19501RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 э
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_19568RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Х
E__inference_sequential_layer_call_and_return_conditional_losses_19835|,-23DENOEЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ00
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 А
(__inference_conv2d_3_layer_call_fn_1956023IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
'__inference_flatten_layer_call_fn_20221T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "џџџџџџџџџH
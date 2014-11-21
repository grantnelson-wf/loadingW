define(function(require) {

    /**
     * Tools for creating and manipulating (3x1) vectors.
     */
    var Vector = {

        /**
         * Creates a new vector.
         * @param  {Number} x  The x component of the vector.
         * @param  {Number} y  The y component of the vector.
         * @param  {Number} z  The z component of the vector.
         * @return  {Array}  The new vector.
         */
        create: function(x, y, z) {
            return [ x, y, z ];
        },
        
        /**
         * Subtracts one vector from another.
         * @param  {Array} vecA  The first vector to subtract from.
         * @param  {Array} vecB  The second vector to subtract.
         * @return  {Array}  The second vector subtracted from the first vector.
         */
        sub: function(vecA, vecB) {
            return [ vecA[0] - vecB[0], vecA[1] - vecB[1], vecA[2] - vecB[2] ];
        },
        
        /**
         * Sums one vector from another.
         * @param  {Array} vecA  The first vector to add to.
         * @param  {Array} vecB  The second vector to add.
         * @return  {Array}  The sum of the two vectors.
         */
        add: function(vecA, vecB) {
            return [ vecA[0] + vecB[0], vecA[1] + vecB[1], vecA[2] + vecB[2] ];
        },
        
        /**
         * Negates the given vector.
         * @param  {Array} vec  The vector to negate.
         * @return  {Array}  The negated vector.
         */
        neg: function(vec) {
            return [ -vec[0], -vec[1], -vec[2] ];
        },
        
        /**
         * Scales the given vector by the given multiplier.
         * @param  {Array} vec  The vector to scale.
         * @param  {Number} scalar  The scalar to multiply each entry by.
         * @return  {Array}  The scaled vector.
         */
        scale: function(vec, scalar) {
            return [ vec[0]*scalar, vec[1]*scalar, vec[2]*scalar ];
        },
        
        /**
         * Normalizes the given vector.
         * @param  {Array} vec  The vector to normalize.
         * @return  {Array}  The normalized array.
         */
        normal: function(vec) {
            var len = Math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
            if (len === 0) {
                return [0, 0, 0];
            } else {
                return [ vec[0]/len, vec[1]/len, vec[2]/len ];
            }
        },
        
        /**
         * Calculates the cross product of the two given vectors.
         * @param  {Array} vecA  The first vector in the cross product.
         * @param  {Array} vecB  The second vector in the cross product.
         * @return  {Array}  The cross product.
         */
        cross: function(vecA, vecB) {
            return [ vecA[1]*vecB[2] - vecA[2]*vecB[1],
                     vecA[2]*vecB[0] - vecA[0]*vecB[2],
                     vecA[0]*vecB[1] - vecA[1]*vecB[0] ];
        },
        
        /**
         * Calculates the dot product of the two given vectors.
         * @param  {Array} vecA  The first vector in the dot product.
         * @param  {Array} vecB  The second vector in the dot product.
         * @return  {Number}  The dot product.
         */
        dot: function(vecA, vecB) {
            return vecA[0]*vecB[0] + vecA[1]*vecB[1] + vecA[2]*vecB[2];
        }
    };

    //======================================================================
    
    /**
     * Tools for creating and manipulating 4x4 matrices.
     */
    var Matrix = {

        /**
         * Creates an identity matrix.
         * @returns  {Array}  The created matrix.
         */
        identity: function() {
            return [ 1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1 ];
        },

        /**
         * Creates a scalar matrix.
         * @param  {Number} scalarX  The value to scale X by. Must be provided.
         * @param  {Number} scalarY  The value to scale Y by. If undefined scalar X will be used.
         * @param  {Number} scalarZ  The value to scale Z by. If undefined scalar Y will be used.
         * @param  {Number} scalarW  The value to scale W by. If undefined scalar Z will be used.
         * @return  {Array}  The created matrix.
         */
        scalar: function(scalarX, scalarY, scalarZ, scalarW) {
            if (scalarY === undefined) {
                scalarY = scalarX;
            }
            if (scalarZ === undefined) {
                scalarZ = scalarY;
            }
            if (scalarW === undefined) {
                scalarW = scalarZ;
            }
            return [ scalarX, 0, 0, 0,
                     0, scalarY, 0, 0,
                     0, 0, scalarZ, 0,
                     0, 0, 0, scalarW ];
        },

        /**
         * Creates a rotation matrix for a X rotation.
         * @param  {Number} angle  The angle in radians to rotate.
         * @returns  {Array}  The created matrix.
         */
        rotateX: function(angle) {
            var c = Math.cos(angle);
            var s = Math.sin(angle);
            return [  1,  0,  0,  0,
                      0,  c, -s,  0,
                      0,  s,  c,  0,
                      0,  0,  0,  1 ];
        },

        /**
         * Creates a rotation matrix for a Y rotation.
         * @param  {Number} angle  The angle in radians to rotate.
         * @returns  {Array}  The created matrix.
         */
        rotateY: function(angle) {
            var c = Math.cos(angle);
            var s = Math.sin(angle);
            return [  c,  0,  s,  0,
                      0,  1,  0,  0,
                     -s,  0,  c,  0,
                      0,  0,  0,  1 ];
        },

        /**
         * Creates a rotation matrix for a Z rotation.
         * @param  {Number} angle  The angle in radians to rotate.
         * @returns  {Array}  The created matrix.
         */
        rotateZ: function(angle) {
            var c = Math.cos(angle);
            var s = Math.sin(angle);
            return [  c, -s,  0,  0,
                      s,  c,  0,  0,
                      0,  0,  1,  0,
                      0,  0,  0,  1 ];
        },

        /**
         * Multiplies the two given matrices.
         * @param  {Array} a  The left matrix in the multiplication.
         * @param  {Array} b  The right matrix in the multiplication.
         * @returns  {Array}  The created matrix.
         */
        mul: function(a, b) {
            // C[x,y] = Sum(A[x,i]*B[y,i]; i=0..3);
            return [ a[ 0]*b[ 0] + a[ 1]*b[ 4] + a[ 2]*b[ 8] + a[ 3]*b[12],
                     a[ 0]*b[ 1] + a[ 1]*b[ 5] + a[ 2]*b[ 9] + a[ 3]*b[13],
                     a[ 0]*b[ 2] + a[ 1]*b[ 6] + a[ 2]*b[10] + a[ 3]*b[14],
                     a[ 0]*b[ 3] + a[ 1]*b[ 7] + a[ 2]*b[11] + a[ 3]*b[15],
                     a[ 4]*b[ 0] + a[ 5]*b[ 4] + a[ 6]*b[ 8] + a[ 7]*b[12],
                     a[ 4]*b[ 1] + a[ 5]*b[ 5] + a[ 6]*b[ 9] + a[ 7]*b[13],
                     a[ 4]*b[ 2] + a[ 5]*b[ 6] + a[ 6]*b[10] + a[ 7]*b[14],
                     a[ 4]*b[ 3] + a[ 5]*b[ 7] + a[ 6]*b[11] + a[ 7]*b[15],
                     a[ 8]*b[ 0] + a[ 9]*b[ 4] + a[10]*b[ 8] + a[11]*b[12],
                     a[ 8]*b[ 1] + a[ 9]*b[ 5] + a[10]*b[ 9] + a[11]*b[13],
                     a[ 8]*b[ 2] + a[ 9]*b[ 6] + a[10]*b[10] + a[11]*b[14],
                     a[ 8]*b[ 3] + a[ 9]*b[ 7] + a[10]*b[11] + a[11]*b[15],
                     a[12]*b[ 0] + a[13]*b[ 4] + a[14]*b[ 8] + a[15]*b[12],
                     a[12]*b[ 1] + a[13]*b[ 5] + a[14]*b[ 9] + a[15]*b[13],
                     a[12]*b[ 2] + a[13]*b[ 6] + a[14]*b[10] + a[15]*b[14],
                     a[12]*b[ 3] + a[13]*b[ 7] + a[14]*b[11] + a[15]*b[15] ];
        },

        /**
         * Multiplies a vertex with the given matrices.
         * @param  {Array} a  The vertex in the multiplication.
         * @param  {Array} b  The matrix in the multiplication.
         * @returns  {Array}  The created matrix.
         */
        vecMul: function(v, m) {
            return [ v[ 0]*m[ 0] + v[ 1]*m[ 4] + v[ 2]*m[ 8] + v[ 3]*m[12],
                     v[ 0]*m[ 1] + v[ 1]*m[ 5] + v[ 2]*m[ 9] + v[ 3]*m[13],
                     v[ 0]*m[ 2] + v[ 1]*m[ 6] + v[ 2]*m[10] + v[ 3]*m[14],
                     v[ 0]*m[ 3] + v[ 1]*m[ 7] + v[ 2]*m[11] + v[ 3]*m[15] ];
        },

        /**
         * Creates a translation matrix.
         * @param  {Number} x  The x offset component.
         * @param  {Number} y  The y offset component.
         * @param  {Number} z  The z offset component.
         * @return  {Array}  The created matrix.
         */
        translate: function(x, y, z) {
            return [  1,  0,  0,  0,
                      0,  1,  0,  0,
                      0,  0,  1,  0,
                      x,  y,  z,  1 ];
        },

        /**
         * Scales the given matrix by the given scalar.
         * @param  {Array} m         The matrix to scale.
         * @param  {Number} scalarX  The value to scale X by. Must be provided.
         * @param  {Number} scalarY  The value to scale Y by. If undefined scalar X will be used.
         * @param  {Number} scalarZ  The value to scale Z by. If undefined scalar Y will be used.
         * @param  {Number} scalarW  The value to scale W by. If undefined scalar Z will be used.
         * @return  {Array}  The scaled matrix.
         */
        scale: function(m, scalarX, scalarY, scalarZ, scalarW) {
            if (scalarY === undefined) {
                scalarY = scalarX;
            }
            if (scalarZ === undefined) {
                scalarZ = scalarY;
            }
            if (scalarW === undefined) {
                scalarW = scalarZ;
            }
            return [  m[ 0]*scalarX, m[ 1]*scalarY, m[ 2]*scalarZ, m[ 3]*scalarW,
                      m[ 4]*scalarX, m[ 5]*scalarY, m[ 6]*scalarZ, m[ 7]*scalarW,
                      m[ 8]*scalarX, m[ 9]*scalarY, m[10]*scalarZ, m[11]*scalarW,
                      m[12]*scalarX, m[13]*scalarY, m[14]*scalarZ, m[15]*scalarW ];
        },

        /**
         * Transposes the given matrix.
         * @param  {Array} m  The matrix to transpose.
         * @return  {Array}  The transposed matrix.
         */
        transpose: function(m) {
            return [  m[ 0], m[ 4], m[ 8], m[12],
                      m[ 1], m[ 5], m[ 9], m[13],
                      m[ 2], m[ 6], m[10], m[14],
                      m[ 3], m[ 7], m[11], m[15] ];
        },

        /**
         * Inverts the given matrix.
         * @param  {Array} m  The matrix to inverse.
         * @return  {Array}  The inverted matrix.
         */
        inverse: function(m) {
            inv = [
                m[ 5]*(m[10]*m[15] - m[11]*m[14]) - m[ 6]*(m[ 9]*m[15] + m[11]*m[13]) + m[ 7]*(m[ 9]*m[14] - m[10]*m[13]),
               -m[ 1]*(m[10]*m[15] + m[11]*m[14]) + m[ 2]*(m[ 9]*m[15] - m[11]*m[13]) - m[ 3]*(m[ 9]*m[14] + m[10]*m[13]),
                m[ 1]*(m[ 6]*m[15] - m[ 7]*m[14]) - m[ 2]*(m[ 5]*m[15] + m[ 7]*m[13]) + m[ 3]*(m[ 5]*m[14] - m[ 6]*m[13]),
               -m[ 1]*(m[ 6]*m[11] + m[ 7]*m[10]) + m[ 2]*(m[ 5]*m[11] - m[ 7]*m[ 9]) - m[ 3]*(m[ 5]*m[10] + m[ 6]*m[ 9]),
               -m[ 4]*(m[10]*m[15] + m[11]*m[14]) + m[ 6]*(m[ 8]*m[15] - m[11]*m[12]) - m[ 7]*(m[ 8]*m[14] + m[10]*m[12]),
                m[ 0]*(m[10]*m[15] - m[11]*m[14]) - m[ 2]*(m[ 8]*m[15] + m[11]*m[12]) + m[ 3]*(m[ 8]*m[14] - m[10]*m[12]),
               -m[ 0]*(m[ 6]*m[15] + m[ 7]*m[14]) + m[ 2]*(m[ 4]*m[15] - m[ 7]*m[12]) - m[ 3]*(m[ 4]*m[14] + m[ 6]*m[12]),
                m[ 0]*(m[ 6]*m[11] - m[ 7]*m[10]) - m[ 2]*(m[ 4]*m[11] + m[ 7]*m[ 8]) + m[ 3]*(m[ 4]*m[10] - m[ 6]*m[ 8]),
                m[ 4]*(m[ 9]*m[15] - m[11]*m[13]) - m[ 5]*(m[ 8]*m[15] + m[11]*m[12]) + m[ 7]*(m[ 8]*m[13] - m[ 9]*m[12]),
               -m[ 0]*(m[ 9]*m[15] + m[11]*m[13]) + m[ 1]*(m[ 8]*m[15] - m[11]*m[12]) - m[ 3]*(m[ 8]*m[13] + m[ 9]*m[12]),
                m[ 0]*(m[ 5]*m[15] - m[ 7]*m[13]) - m[ 1]*(m[ 4]*m[15] + m[ 7]*m[12]) + m[ 3]*(m[ 4]*m[13] - m[ 5]*m[12]),
               -m[ 0]*(m[ 5]*m[11] + m[ 7]*m[ 9]) + m[ 1]*(m[ 4]*m[11] - m[ 7]*m[ 8]) - m[ 3]*(m[ 4]*m[ 9] + m[ 5]*m[ 8]),
               -m[ 4]*(m[ 9]*m[14] + m[10]*m[13]) + m[ 5]*(m[ 8]*m[14] - m[10]*m[12]) - m[ 6]*(m[ 8]*m[13] + m[ 9]*m[12]),
                m[ 0]*(m[ 9]*m[14] - m[10]*m[13]) - m[ 1]*(m[ 8]*m[14] + m[10]*m[12]) + m[ 2]*(m[ 8]*m[13] - m[ 9]*m[12]),
               -m[ 0]*(m[ 5]*m[14] + m[ 6]*m[13]) + m[ 1]*(m[ 4]*m[14] - m[ 6]*m[12]) - m[ 2]*(m[ 4]*m[13] + m[ 5]*m[12]),
                m[ 0]*(m[ 5]*m[10] - m[ 6]*m[ 9]) - m[ 1]*(m[ 4]*m[10] + m[ 6]*m[ 8]) + m[ 2]*(m[ 4]*m[ 9] - m[ 5]*m[ 8])
            ];

            det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
            if (det === 0) {
                return Matrix.identity();
            }
            return Matrix.scale(inv, 1/det);
        },

        /**
         * Creates a rotation matrix with the given Euler angles.
         * @note  The order of rotation is yaw, pitch, then roll.
         * @param  {Number} yaw   The yaw Euler angle in radians.
         * @param  {Number} pitch The pitch Euler angle in radians.
         * @param  {Number} roll  The roll Euler angle in radians.
         * @return  {Array}  The rotation matrix.
         */
        euler: function(yaw, pitch, roll) {
            return this.mul(
                    this.mul(
                        this.rotateX(yaw),
                        this.rotateY(pitch)),
                    this.rotateZ(roll));

            // TODO: Need to check the following as a simplification.
            // Euler = RotX(yaw) * RotY(pitch) * RotZ(roll)
            //var cY = Math.cos(yaw),   sY = Math.sin(yaw);
            //var cP = Math.cos(pitch), sP = Math.sin(pitch);
            //var cR = Math.cos(roll),  sR = Math.sin(roll);
            //return [     cP*cR,           -cP*sR,           sP,  0,
            //         cY*sR + sY*sP*cR, cY*cR - sY*sP*sR, -sY*cP, 0,
            //         sY*sR - cY*sP*cR, sY*cR + cY*sP*sR,  cY*cP, 0,
            //               0,                0,             0,   1 ];
        },

        /**
         * Creates an orthographic projection matrix.
         * @param  {Number} left    The left side of the projection.
         * @param  {Number} right   The right side of the projection.
         * @param  {Number} bottom  The bottom side of the projection.
         * @param  {Number} top     The top side of the projection.
         * @param  {Number} near    The near side of the projection.
         * @param  {Number} far     The far side of the projection.
         * @return  {Array}  The created matrix.
         */
        orthographic: function(left, right, bottom, top, near, far) {
            var xx = 2.0/(right-left);
            var yy = 2.0/(top-bottom);
            var zz = 2.0/(far-near);
            var wx = -(left+right)/(right-left);
            var wy = -(top+bottom)/(top-bottom);
            var wz = (far+near)/(far-near);
            return [ xx,  0,  0, wx,
                      0, yy,  0, wy,
                      0,  0, zz, wz,
                      0,  0,  0,  1 ];
        },
        
        /**
         * Creates a focused camera matrix.
         * @param  {Number} targetX  The x component the camera targets.
         * @param  {Number} targetY  The y component the camera targets.
         * @param  {Number} targetZ  The z component the camera targets.
         * @param  {Number} upX      The x component of the upward direction for the camera.
         * @param  {Number} upY      The y component of the upward direction for the camera.
         * @param  {Number} upZ      The z component of the upward direction for the camera.
         * @param  {Number} posX     The x component of the camera.
         * @param  {Number} posY     The y component of the camera.
         * @param  {Number} posZ     The z component of the camera.
         * @return  {Array}  The created focused camera matrix.
         */
        lookat: function(targetX, targetY, targetZ, upX, upY, upZ, posX, posY, posZ) {
            var target = Vector.create(targetX, targetY, targetZ);
            var up = Vector.create(upX, upY, upZ);
            var pos = Vector.create(posX, posY, posZ);
            var zaxis = Vector.normal(Vector.sub(target, pos));
            var xaxis = Vector.normal(Vector.cross(up, zaxis));
            var yaxis = Vector.cross(zaxis, xaxis);
            var tx = -Vector.dot(xaxis, pos);
            var ty = -Vector.dot(yaxis, pos);
            var tz = -Vector.dot(zaxis, pos);
            return [ xaxis[0], yaxis[0], zaxis[0], 0,
                     xaxis[1], yaxis[1], zaxis[1], 0,
                     xaxis[2], yaxis[2], zaxis[2], 0,
                     tx,       ty,       tz,       1 ];
        },

        /**
         * Creates an perspective projection matrix.
         * @param  {Number} fov     The angle in radians for the vertical field of view.
         * @param  {Number} aspect  The aspect ratio of horizontal over vertical.
         * @param  {Number} near    The near side of the frustum.
         * @param  {Number} far     The far side of the frustum.
         * @returns  {Array}  The created matrix.
         */
        perspective: function(fov, aspect, near, far) {
            var yy = 1.0/Math.tan(fov*0.5);
            var xx = yy/aspect;
            var zz = far/(far-near);
            var zw = -far*near/(far-near);
            return [ xx,  0,  0,  0,
                      0, yy,  0,  0,
                      0,  0, zz,  1,
                      0,  0, zw,  0 ];
        },

        /**
         * Converts the given matrix into a string.
         * @param  {Array} mat        The matrix to convert to a string.
         * @param  {String} [indent]  Optional indent to apply to new lines.
         * @returns  {String}  The string for the matrix.
         */
        toString: function(mat, indent) {
            indent = indent || '';
            return     '['+mat[ 0]+', '+mat[ 1]+', '+mat[ 2]+', '+mat[ 3]+',\n'+
                indent+' '+mat[ 4]+', '+mat[ 5]+', '+mat[ 6]+', '+mat[ 7]+',\n'+
                indent+' '+mat[ 8]+', '+mat[ 9]+', '+mat[10]+', '+mat[11]+',\n'+
                indent+' '+mat[12]+', '+mat[13]+', '+mat[14]+', '+mat[15]+']';
        }

    };

    //======================================================================
    
    /**
     * [Shader description]
     */
    function Shader() {
        
        /// The graphical object.
        this._gl = null;

        /// The program shader object.
        this._program = null;

        this._posAttrLoc = null;  //
        this._normAttrLoc = null; //

        this._objMatLoc = -1;   //
        this._viewMatLoc = -1;  //
        this._projMatLoc = -1;  //
        this._lightVecLoc = -1; //
        this._lightClrLoc = -1; //
        this._darkClrLoc = -1;  //
    }

    /**
     * The vertex shader program.
     * @type {String}
     */
    Shader.prototype._vsSource =
        'uniform mat4 objMat;                                       \n'+
        'uniform mat4 viewMat;                                      \n'+
        'uniform mat4 projMat;                                      \n'+
        'uniform vec3 lightVec;                                     \n'+
        '                                                           \n'+
        'attribute vec3 posAttr;                                    \n'+
        'attribute vec3 normAttr;                                   \n'+
        '                                                           \n'+
        'varying vec3 normal;                                       \n'+
        'varying vec3 litVec;                                       \n'+
        '                                                           \n'+
        'void main()                                                \n'+
        '{                                                          \n'+
        '  normal = (objMat*vec4(normAttr, 0.0)).xyz;               \n'+
        '  litVec = normalize((viewMat*vec4(lightVec, 0.0)).xyz);   \n'+
        '  gl_Position = projMat*viewMat*objMat*vec4(posAttr, 1.0); \n'+
        '}                                                          \n';

    /**
     * The fragment shader program.
     * @type {String}
     */
    Shader.prototype._fsSource =
        'precision mediump float;                         \n'+
        '                                                 \n'+
        'uniform vec4 lightClr;                           \n'+
        'uniform vec4 darkClr;                            \n'+
        '                                                 \n'+
        'varying vec3 normal;                             \n'+
        'varying vec3 litVec;                             \n'+
        '                                                 \n'+
        'void main()                                      \n'+
        '{                                                \n'+
        '   vec3 norm = normalize(normal);                \n'+
        '   float diffuse = max(dot(norm, litVec), 0.0);  \n'+
        '   float shade = 1.0 - clamp(diffuse, 0.0, 1.0); \n'+
        '   gl_FragColor = mix(lightClr, darkClr, shade); \n'+
        '}                                                \n';
    
    /**
     * [setup description]
     * @param  {[type]} gl [description]
     * @return {[type]}    [description]
     */
    Shader.prototype.setup = function(gl) {
        this._gl = gl;

        // Compile shaders.
        var vsShader = this._compileShader(this._vsSource, gl.VERTEX_SHADER);
        if (!vsShader) {
            console.log('Failed to compile VS shader.');
            return null;
        }
       
        var fsShader = this._compileShader(this._fsSource, gl.FRAGMENT_SHADER);
        if (!fsShader) {
            console.log('Failed to compile FS shader.');
            return null;
        }
       
        // Link shaders to program.
        this._program = gl.createProgram();
        this._gl.attachShader(this._program, vsShader);
        this._gl.attachShader(this._program, fsShader);
        this._gl.linkProgram(this._program);
        if (!this._gl.getProgramParameter(this._program, gl.LINK_STATUS)) {
           console.log('Could not link shaders.');
           return false;
        }

        // Get attribute accessors.
        this._posAttrLoc  = this._gl.getAttribLocation(this._program, 'posAttr');
        this._normAttrLoc = this._gl.getAttribLocation(this._program, 'normAttr');

        // Get uniform variables accessors.
        this._objMatLoc   = this._gl.getUniformLocation(this._program, 'objMat');
        this._viewMatLoc  = this._gl.getUniformLocation(this._program, 'viewMat');
        this._projMatLoc  = this._gl.getUniformLocation(this._program, 'projMat');
        this._lightVecLoc = this._gl.getUniformLocation(this._program, 'lightVec');
        this._lightClrLoc = this._gl.getUniformLocation(this._program, 'lightClr');
        this._darkClrLoc  = this._gl.getUniformLocation(this._program, 'darkClr');

        return true;
    };
   
    /**
     * This compiles a shader.
     * @param  {String} source      The source string for the shader.
     * @param  {Number} shaderType  The type of shader to compile.
     * @returns  {Object}  The compiled shader.
     */
    Shader.prototype._compileShader = function(source, shaderType) {
       var shader = this._gl.createShader(shaderType);
       this._gl.shaderSource(shader, source);
       this._gl.compileShader(shader);
       if (!this._gl.getShaderParameter(shader, this._gl.COMPILE_STATUS)) {
          console.log(this._gl.getShaderInfoLog(shader));
          return null;
       }
       return shader;
    };

    /**
     * 
     */
    Shader.prototype.use = function() {
        this._gl.useProgram(this._program);
    };

    ///
    Shader.prototype.getPosAttrLoc = function() {
        return this._posAttrLoc;
    };

    ///
    Shader.prototype.getNormAttrLoc = function() {
        return this._normAttrLoc;
    };

    ///
    Shader.prototype.setObjectMatrix = function(matrix) {
        this._gl.uniformMatrix4fv(this._objMatLoc, false, new Float32Array(matrix));
    };

    ///
    Shader.prototype.setViewMatrix = function(matrix) {
        this._gl.uniformMatrix4fv(this._viewMatLoc, false, new Float32Array(matrix));
    };

    ///
    Shader.prototype.setProjectionMatrix = function(matrix) {
        this._gl.uniformMatrix4fv(this._projMatLoc, false, new Float32Array(matrix));
    };

    ///
    Shader.prototype.setLightVector = function(x, y, z) {
        this._gl.uniform3f(this._lightVecLoc, x, y, z);
    };

    ///
    Shader.prototype.setLightColor = function(red, green, blue, alpha) {
        this._gl.uniform4f(this._lightClrLoc, red, green, blue, alpha);
    };

    ///
    Shader.prototype.setDarkColor = function(red, green, blue, alpha) {
        this._gl.uniform4f(this._darkClrLoc, red, green, blue, alpha);
    };

    //======================================================================
    
    function Shape(gl, vertexBuf, indexObjs) {

        this._gl = gl;
        this._vertexBuf = vertexBuf;
        this._indexObjs = indexObjs;
        this._posAttrLoc = null;
        this._normAttrLoc = null;
    }

    Shape.prototype.setAttrs = function(posAttrLoc, normAttrLoc) {
        this._posAttrLoc  = posAttrLoc;
        this._normAttrLoc = normAttrLoc;
    };

    Shape.prototype.bind = function() {
        var stride = 6*Float32Array.BYTES_PER_ELEMENT;
        var offset = 3*Float32Array.BYTES_PER_ELEMENT;
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, this._vertexBuf);
        this._gl.enableVertexAttribArray(this._posAttrLoc);
        this._gl.enableVertexAttribArray(this._normAttrLoc);
        this._gl.vertexAttribPointer(this._posAttrLoc,  3, this._gl.FLOAT, false, stride, 0);
        this._gl.vertexAttribPointer(this._normAttrLoc, 3, this._gl.FLOAT, false, stride, offset);
    };

    Shape.prototype.unbind = function() {
        this._gl.disableVertexAttribArray(this._posAttrLoc);
        this._gl.disableVertexAttribArray(this._normAttrLoc);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, null);
    };

    Shape.prototype.draw = function() {
        for (var i = this._indexObjs.length-1; i >= 0; i--) {
            var indexObj = this._indexObjs[i];
            this._gl.bindBuffer(this._gl.ELEMENT_ARRAY_BUFFER, indexObj.buffer);
            this._gl.drawElements(indexObj.type, indexObj.count, this._gl.UNSIGNED_SHORT, 0);
        }
    };

    //======================================================================

    function ShapeBuilder() {

        this.vertices = [];

        this.currentFan = [];

        this.fans = [];
    }

    ShapeBuilder.prototype.build = function(gl) {
        //                              I___J
        //                              |   |
        //                             |   |
        //                            |   |
        //  F___G         M___N      |   |
        //  |   |         |   |     |   |
        //   |   |   |B    |   |   |   |
        //    |   | | |     |   | |   |
        //     |  A'   |C    |  H'   |
        //      |_____|       |_____|
        //      E     D       L     K
        this._addPoly([
            36, 83, 46, 57, 57, 90, 48, 115, // A, B, C, D
            24, 115, 0, 47, 22,47, ]); // E, F, G
        this._addPoly([
            82, 83, 111, 0, 133, 0, 95, 115, // H, I, J, K
            72, 115, 48, 47, 70, 47 ]); // L, M, N


        // Create buffers and shape object.
        var vertexBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(this.vertices), gl.STATIC_DRAW);

        var indexObjs = [];
        for (var i = this.fans.length - 1; i >= 0; i--) {
            var indices = this.fans[i];
            var indexBuf  = gl.createBuffer();
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuf);
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
            indexObjs.push({
                type:   gl.TRIANGLE_FAN,
                count:  indices.length,
                buffer: indexBuf
            });
        }

        return new Shape(gl, vertexBuf, indexObjs);
    };

    ShapeBuilder.prototype._closeTriFan = function() {
        if (this.currentFan.length < 3) {
            throw 'Error: Must have at least 3 indices in the triangle fan.';
        }
        this.fans.push(this.currentFan);
        this.currentFan = [];
    };

    /**
     * Adds a vertex to the shape.
     * @param  {Number} px  The x component of the position.
     * @param  {Number} py  The y component of the position.
     * @param  {Number} pz  The z component of the position.
     * @param  {Number} nx  The x component of the normal.
     * @param  {Number} ny  The y component of the normal.
     * @param  {Number} nz  The z component of the normal.
     */
    ShapeBuilder.prototype._addVec = function(px, py, pz, nx, ny, nz) {
        // Add index to current triangle fan.
        var index = this.vertices.length / 6;
        this.currentFan.push(Number(index));

        // Add vertex to list.
        var scalar = 1/133;
        px = Number(px)*scalar-0.5;
        py = Number(py)*scalar-0.5;
        pz = Number(pz)*scalar-0.5;
        this.vertices.push(Number(px), Number(py), Number(pz),
                           Number(nx), Number(ny), Number(nz));
    };
    
    /**
     * Created an extended polygon and adds it to the shape builder.
     * Expects the polygon to be fan-able fill at the first point.
     * @param  {Array} poly  The x, y tuples for the polygon to add and extend.
     */
    ShapeBuilder.prototype._addPoly = function(poly) {
        var i, index;
        var count = poly.length/2;
        if (count < 3) {
            throw 'Error: Must have at least 3 vertices in the polygon.';
        }

        // Add front face (uses fan for triangles).
        this._addVec(poly[0], poly[1], 77, 0, 0, 1);
        for (i = 1, j = count - 1; i < count; i++, j--) {
            this._addVec(poly[j*2], poly[j*2+1], 77, 0, 0, 1);
        }
        this._closeTriFan();

        // Add back face (uses fan for triangles).
        for (i = 0; i < count; i++) {
            this._addVec(poly[i*2], poly[i*2+1], 55, 0, 0, -1);
        }
        this._closeTriFan();

        // Add joining faces.
        var x1 = poly[count*2-2], y1 = poly[count*2-1];
        for (i = 0; i < count; i++) {
            var x2 = poly[i*2],   y2 = poly[i*2+1];
            var dx = x2-x1,       dy = y2-y1;
            var len = Math.sqrt(dx*dx + dy*dy);
            var nx = dy/len, ny = -dx/len;

            this._addVec(x1, y1, 77, nx, ny, 0);
            this._addVec(x2, y2, 77, nx, ny, 0);
            this._addVec(x2, y2, 55, nx, ny, 0);
            this._addVec(x1, y1, 55, nx, ny, 0);
            this._closeTriFan();

            x1 = x2;
            y1 = y2;
        }
    };
    
    //======================================================================

    /**
     * The wobble mover rotates at a rate with a cosine multiplier.
     */
    function Mover() {
        
        this.yawSpeed = 0.6; // The yaw speed in radians.
        this.pitchSpeed = 0.8; //The pitch speed in radians.
        this.rollSpeed = 1.0; // The roll speed in radians.

        this.yawOffset = 0.0; // The yaw offset in radians.
        this.pitchOffset = 0.0; // The pitch offset in radians.
        this.rollOffset = 0.0; // The roll offset in radians.
        
        this.deltaYaw = 0.2; // The maximum yaw in radians.
        this.deltaPitch = 0.4; // The maximum pitch in radians.
        this.deltaRoll = 0.2; // The maximum roll in radians.
        
        this.initYaw = Math.PI; // The initial yaw in radians.
        this.initPitch = 0.0; // The initial pitch in radians.
        this.initRoll = 0.0; // The initial roll in radians.
        
        /**
         * The start time in milliseconds.
         * @type {Number}
         */
        this.startTime = (new Date()).getTime();
    }

    /**
     * Updates the mover.
     */
    Mover.prototype.getMatrix = function() {
        var curTime = (new Date()).getTime();
        var dt = (curTime - this.startTime)/1000;
        var yaw   = Math.cos(dt*this.yawSpeed   + this.yawOffset)  *this.deltaYaw   + this.initYaw;
        var pitch = Math.cos(dt*this.pitchSpeed + this.pitchOffset)*this.deltaPitch + this.initPitch;
        var roll  = Math.cos(dt*this.rollSpeed  + this.rollOffset) *this.deltaRoll  + this.initRoll;
        return Matrix.euler(yaw, pitch, roll);
    };

    //======================================================================
    
    function Graphics3D() {
        
        /// The graphical object.
        this._gl = null;

        /// The shader manager.
        this._shader = null;

        /// The shape manager.
        this._shape = null;

        /// The mover for the object matrix.
        this._mover = null;

    }

    /**
     * [_setupGraphics description]
     * @return {[type]} [description]
     */
    Graphics3D.prototype.setup = function(gl) {
        this._gl = gl;
        this._shader = new Shader();
        if (!this._shader.setup(this._gl)) {
            return false;
        }
        this._shader.use();
        this._shader.setViewMatrix(this._createViewMatrix());
        this._shader.setProjectionMatrix(this._createProjMatrix());
        this._shader.setLightVector(0.5, 0.5, -1.0);

        shapeBuilder = new ShapeBuilder();
        this._shape = shapeBuilder.build(gl);
        this._shape.setAttrs(this._shader.getPosAttrLoc(), this._shader.getNormAttrLoc());
        this._shape.bind();

        this._mover = new Mover();

        this._gl.clearColor(0.0, 0.0, 0.0, 0.0);
        this._gl.clearDepth(1.0);
        this._gl.disable(this._gl.CULL_FACE);
        this._gl.enable(this._gl.DEPTH_TEST);
        return true;
    };

    /**
     * [resize description]
     * @param  {[type]} width  [description]
     * @param  {[type]} height [description]
     * @return {[type]}        [description]
     */
    Graphics3D.prototype.resize = function(width, height) {
        this._gl.viewport(0, 0, width, height);

        // Update the aspect ratio.
        if (this._shader !== null) {
            this._shader.setProjectionMatrix(this._createProjMatrix());
        }
    };
    
    /**
     * [_render description]
     * @return {[type]} [description]
     */
    Graphics3D.prototype.render = function() {
        this._gl.clear(this._gl.COLOR_BUFFER_BIT|this._gl.DEPTH_BUFFER_BIT);
        var mat = this._mover.getMatrix();

        var shadowMat = Matrix.scale(Matrix.mul(mat, Matrix.translate(0.1, -0.1, 0.0)), 1.6, 1.6, 0.0, 1.0);
        this._shader.setObjectMatrix(shadowMat);
        this._shader.setLightColor(0.01, 0.02, 0.0, 0.17);
        this._shader.setDarkColor(0.01, 0.02, 0.0, 0.17);
        this._shape.draw();

        var shapeMat = Matrix.mul(mat, Matrix.translate(0.0, 0.0, -1.0));
        this._shader.setObjectMatrix(shapeMat);
        this._shader.setLightColor(0.44, 0.77, 0.04, 1.0);
        this._shader.setDarkColor(0.22, 0.38, 0.02, 1.0);
        this._shape.draw();
    };

    /**
     * [_getViewMatrix description]
     * @return {[type]} [description]
     */
    Graphics3D.prototype._createViewMatrix = function() {
        return Matrix.lookat(0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, -2.5);
    };

    /**
     * [_getProjectionMatrix description]
     * @return {[type]} [description]
     */
    Graphics3D.prototype._createProjMatrix = function() {
        var aspect = this._gl.drawingBufferWidth / this._gl.drawingBufferHeight;
        return Matrix.perspective(Math.PI/3.0, aspect, 1.0, -1.0);
    };

    //======================================================================

    /**
     * Creates the graphical driver.
     */
    function LoadingW() {

        /// The div element this loading W is tied to.
        this._div = null;
    
        /// The canvas element.
        this._canvas = null;

        /// The graphics manager.
        this._graphics = null;

    }
    
    /**
     * Sets up the loading W.
     * @param  {String} divId  The div element id for the loading W to draw on.
     */
    LoadingW.prototype.setup = function(divId) {
        this._div = document.getElementById(divId);
        if (!this._div) {
            console.log('Failed to retrieve the <div> element.');
            return false;
        }

        this._canvas = document.createElement('canvas');
        this._canvas.id = 'LoadingW';
        this._div.appendChild(this._canvas);

        var gl = null;
        try {
            gl = this._canvas.getContext('webgl');
        }
        catch(ex) {
           console.log('Error getting WebGL context: '+err.message);
           return false;
        }

        if (gl) {
            this._graphics = new Graphics3D();
            if (!this._graphics.setup(gl)) {
                console.log('Failed to setup graphics.');
                return false;
            }
        } else {

           // TODO: Do fall back.

           console.log('Failed to get the rendering context for WebGL.');
           return false;
        }

        // Setup automatic resizing.
        var self = this;
        this._div.addEventListener('resize', function() {
            self._resize();
        });
        this._resize();
        return true;
    };

    /**
     * This resizes the canvas.
     */
    LoadingW.prototype.show = function() {
        this._canvas.style.visibility = 'visible';

        // Start update loop
        this._update();
    };

    /**
     * This resizes the canvas.
     */
    LoadingW.prototype.hide = function() {
        this._canvas.style.visibility = 'hidden';
        // Update loop will end because 'visibility' is not set to 'visible'.
    };


    /**
     * This resizes the canvas.
     */
    LoadingW.prototype._resize = function() {
        var width  = this._div.offsetWidth;
        var height = this._div.offsetHeight;

        if ((this._canvas.width !== width) || (this._canvas.height !== height)) {
            this._canvas.width  = width;
            this._canvas.height = height;
            if (this._graphics !== null) {
                this._graphics.resize(width, height);
            }
        }
    };
    
    /**
     * This updates the rendering and continues updating the rendering
     * until the selected item stops the update.
     */
    LoadingW.prototype._update = function() {
        if (this._canvas.style.visibility !== 'visible') {
            return;
        } else if (this._graphics === null) {
            return;
        }

        this._graphics.render();

        var self = this;
        requestAnimationFrame(function() {
            self._update();
        });
    };
 
    return LoadingW;
});

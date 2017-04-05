#version 330 core
// interplotade values from the vertex shader
in vec3 Position_worldspace;
in vec3 Normal_cameraspace;
in vec3 LightDirection_cameraspace;
in vec3 EyeDirection_cameraspace;
// output data
out vec4 color;
// values that stay constant for the whole mesh
uniform vec3 LightPosition_worldspace;
uniform vec3 LightColor;
uniform float LightPower;
uniform vec3 MaterialDiffuseColor;
uniform vec3 MaterialAmbientColorCoeffs;
uniform vec3 MaterialSpecularColor;

void main() {
	vec3 n = normalize(Normal_cameraspace);
	vec3 l = normalize(LightDirection_cameraspace);
	// eye vector (towards the camera)
	vec3 E = normalize(EyeDirection_cameraspace);
	// direction in which the triangle reflects the light
	vec3 R = reflect(-l, n);
	float cosTheta = clamp(dot(n, l), 0, 1);
	float cosAlpha = clamp(dot(E, R), 0, 1);
	vec3 MaterialAmbientColor = MaterialAmbientColorCoeffs*MaterialDiffuseColor;
	float distance = length(LightPosition_worldspace - Position_worldspace);
	color.xyz =
		MaterialAmbientColor +
		MaterialDiffuseColor*LightColor*LightPower*cosTheta/(distance*distance) +
		MaterialSpecularColor*LightColor*LightPower*pow(cosAlpha, 5)/(distance*distance);
	color.a = 1;
}

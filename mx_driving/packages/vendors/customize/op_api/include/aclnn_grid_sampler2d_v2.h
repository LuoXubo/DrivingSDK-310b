
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GRID_SAMPLER2D_V2_H_
#define ACLNN_GRID_SAMPLER2D_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGridSampler2dV2GetWorkspaceSize
 * parameters :
 * xTrans : required
 * grid : required
 * interpolationMode : required
 * paddingMode : required
 * alignCorners : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGridSampler2dV2GetWorkspaceSize(
    const aclTensor *xTrans,
    const aclTensor *grid,
    int64_t interpolationMode,
    int64_t paddingMode,
    bool alignCorners,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGridSampler2dV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGridSampler2dV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

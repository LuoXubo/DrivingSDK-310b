
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FAKE_TENSOR_QUANT_WITH_AXIS_H_
#define ACLNN_FAKE_TENSOR_QUANT_WITH_AXIS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnFakeTensorQuantWithAxisGetWorkspaceSize
 * parameters :
 * inputs : required
 * amax : required
 * axis : required
 * numBits : required
 * isUnsigned : required
 * narrowRange : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFakeTensorQuantWithAxisGetWorkspaceSize(
    const aclTensor *inputs,
    const aclTensor *amax,
    int64_t axis,
    int64_t numBits,
    bool isUnsigned,
    bool narrowRange,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnFakeTensorQuantWithAxis
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFakeTensorQuantWithAxis(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

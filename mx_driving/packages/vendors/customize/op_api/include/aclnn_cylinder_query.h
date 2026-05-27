
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CYLINDER_QUERY_H_
#define ACLNN_CYLINDER_QUERY_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnCylinderQueryGetWorkspaceSize
 * parameters :
 * newXyz : required
 * xyz : required
 * rot : required
 * originIndex : required
 * batchSize : required
 * pointCloudSize : required
 * queryPointSize : required
 * radius : required
 * hmin : required
 * hmax : required
 * nsample : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCylinderQueryGetWorkspaceSize(
    const aclTensor *newXyz,
    const aclTensor *xyz,
    const aclTensor *rot,
    const aclTensor *originIndex,
    int64_t batchSize,
    int64_t pointCloudSize,
    int64_t queryPointSize,
    double radius,
    double hmin,
    double hmax,
    int64_t nsample,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnCylinderQuery
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCylinderQuery(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

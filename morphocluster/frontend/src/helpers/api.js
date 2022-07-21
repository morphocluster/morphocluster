import axios from "axios";

export function getDatasets() {
    return axios.get(`/api/datasets`, { params: {} })
        .then(response => {
            return response.data;
        });
}

export function getNode(dataset_id, project_id, node_id) {
    return axios.get(`/api/datasets/${dataset_id}/projects/${project_id}/nodes/${node_id}`)
        .then(response => {
            return response.data;
        });
}

export function patchNode(dataset_id, project_id, node_id, data) {
    return axios.patch(`/api/datasets/${dataset_id}/projects/${project_id}/nodes/${node_id}`, data)
        .then(response => {
            return response.data;
        });
}

/**
 * Get the next unapproved node for a subtree rooted at node_id.
 *
 * @param params {leaf: bool}
 */
export function getNextUnapprovedNode(dataset_id, project_id, node_id, params = null) {
    return axios.get(`/api/datasets/${dataset_id}/projects/${project_id}/nodes/${node_id}/next`, { params })
        .then(response => {
            return response.data;
        });
}

/**
 * Get the next unfilled node for a subtree rooted at node_id.
 *
 * @param params {leaf: bool, preferred_first: bool, order_by: string}
 */
export function getNextUnfilledNode(dataset_id, project_id, node_id, params = null) {
    return axios.get(`/api/datasets/${dataset_id}/projects/${project_id}/nodes/${node_id}/next_unfilled`, { params })
        .then(response => {
            return response.data;
        });
}

/**
 * Get the sorting progress for a subtree rooted at node_id.
 *
 * @param params {log: string}
 */
export function getNodeProgress(dataset_id, project_id, node_id, params = null) {
    return axios.get(`/api/datasets/${dataset_id}/projects/${project_id}/nodes/${node_id}/progress`, { params })
        .then(response => {
            return response.data;
        });
}

/**
 * Get recommended objects for node_id.
 *
 * @param params {max_n: int}
 */
export function getNodeRecommendedObjects(dataset_id, project_id, node_id, params = null) {
    return axios.get(`/api/datasets/${dataset_id}/projects/${project_id}/nodes/${node_id}/recommended_objects`, { params })
        .then(response => {
            return response.data;
        });
}

export function mergeNodeInto(dataset_id, project_id, node_id, dest_node_id) {
    const data = { dest_node_id };
    console.log(data)
    return axios.post(`/api/datasets/${dataset_id}/projects/${project_id}/nodes/${node_id}/merge_into`, data);
}

// Project

export function getProjects(dataset_id, include_progress = false) {
    return axios.get(`/api/datasets/${dataset_id}/projects`, { params: { include_progress } })
        .then(response => {
            return response.data;
        });
}

export function getProject(dataset_id, project_id, include_progress = false) {
    return axios.get(`/api/datasets/${dataset_id}/projects/${project_id}`, { params: { include_progress } })
        .then(response => {
            return response.data;
        });
}

export function saveProject(dataset_id, project_id) {
    return axios.post(`/api/datasets/${dataset_id}/projects/${project_id}/save`)
        .then(response => {
            return response.data;
        });
}

export function nodeAdoptMembers(dataset_id, project_id, node_id, members) {
    if (!Array.isArray(members)) {
        members = [members];
    }

    return axios.post(`/api/datasets/${dataset_id}/projects/${project_id}/nodes/${node_id}/adopt_members`, { members: members });
}

export function nodeAcceptRecommendations(dataset_id, project_id, node_id, request_id, rejected_members, last_page, log_data = null) {
    return axios.post(`/api/datasets/${dataset_id}/projects/${project_id}/nodes/${node_id}/accept_recommended_objects`,
        { request_id, rejected_members, last_page, log_data });
}

export function getUnfilledNodes(dataset_id, project_id) {
    return axios.get(`/api/datasets/${dataset_id}/projects/${project_id}/unfilled_nodes`).then(response => {
        return response.data;
    });
}

export function log(action, node_id = null, reverse_action = null, data = null) {
    return axios.post(`/api/log`,
        { action, node_id, reverse_action, data });
}
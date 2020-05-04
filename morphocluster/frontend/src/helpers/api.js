import axios from "axios";

export function getNode(project_id, node_id) {
    return axios.get(`/api/projects/${project_id}/nodes/${node_id}`)
        .then(response => {
            return response.data;
        });
}

export function patchNode(project_id, node_id, data) {
    return axios.patch(`/api/projects/${project_id}/nodes/${node_id}`, data)
        .then(response => {
            return response.data;
        });
}

export function getNextUnapprovedNode(project_id, node_id = null, leaf = false) {
    var url = node_id ? `/api/projects/${project_id}/nodes/${node_id}/next_unapproved` : `/api/projects/${project_id}/next_unapproved`;
    return axios.get(url, { params: { leaf } })
        .then(response => {
            return response.data;
        });
}

export function getNextUnfilledNode(project_id, node_id, leaf = false, preferred_first = false) {
    var url = node_id ? `/api/projects/${project_id}/nodes/${node_id}/next_unfilled` : `/api/projects/${project_id}/next_unfilled`;
    return axios.get(url, { params: { leaf, preferred_first } })
        .then(response => {
            return response.data;
        });
}

export function getNodeProgress(project_id, node_id, log = null) {
    var params = {}
    if (log !== null) {
        params.log = log;
    }
    return axios.get(`/api/projects/${project_id}/nodes/${node_id}/progress`, { params })
        .then(response => {
            return response.data;
        });
}

export function getNodeRecommendedObjects(project_id, node_id, max_n = null) {
    var params = {}
    if (max_n !== null) {
        params.max_n = max_n;
    }
    return axios.get(`/api/projects/${project_id}/nodes/${node_id}/recommended_objects`, { params })
        .then(response => {
            return response.data;
        });
}

export function mergeNodeInto(project_id, node_id, dest_node_id) {
    const data = { dest_node_id };
    console.log(data)
    return axios.post(`/api/projects/${project_id}/nodes/${node_id}/merge_into`, data);
}

// Project

export function getProject(project_id, include_progress = false, include_dataset = false) {
    return axios.get(`/api/projects/${project_id}`, { params: { include_progress, include_dataset } })
        .then(response => {
            return response.data;
        });
}

export function saveProject(project_id) {
    return axios.post(`/api/projects/${project_id}/save`)
        .then(response => {
            return response.data;
        });
}

export function nodeAdoptMembers(project_id, node_id, members) {
    if (!Array.isArray(members)) {
        members = [members];
    }

    return axios.post(`/api/projects/${project_id}/nodes/${node_id}/adopt_members`, { members: members });
}

export function nodeAcceptRecommendations(project_id, node_id, request_id, rejected_members, last_page, log_data = null) {
    return axios.post(`/api/projects/${project_id}/nodes/${node_id}/accept_recommended_objects`,
        { request_id, rejected_members, last_page, log_data });
}

export function getUnfilledNodes(project_id) {
    return axios.get(`/api/projects/${project_id}/unfilled_nodes`).then(response => {
        return response.data;
    });
}

export function log(action, node_id = null, reverse_action = null, data = null) {
    return axios.post(`/api/log`,
        { action, node_id, reverse_action, data });
}

// Dataset

export function datasetsGetAll() {
    return axios.get(`/api/datasets`)
        .then(response => {
            return response.data;
        });
}

export function createDataset(properties = null) {
    return axios.post(`/api/datasets`)
        .then(response => {
            return response.data;
        });
}

export function getDataset(dataset_id) {
    return axios.get(`/api/datasets/${dataset_id}`)
        .then(response => {
            return response.data;
        });
}
export function datasetGetProjects(dataset_id, include_progress = false) {
    return axios.get(`/api/datasets/${dataset_id}/projects`, { params: { include_progress } })
        .then(response => {
            return response.data;
        });
}
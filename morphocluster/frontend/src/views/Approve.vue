<template>
    <div id="approve">
        <nav class="navbar navbar-expand-lg navbar-light bg-dark">
            <router-link class="navbar-brand text-light" to="/">MorphoCluster</router-link>
            <div class="collapse navbar-collapse">
                <ul class="navbar-nav mr-auto">
                    <li class="nav-item nav-link text-light" v-if="project">
                        {{project.name}}
                    </li>
                    <li class="nav-item nav-link  text-light">
                        Approve
                    </li>
                    <li class="nav-item  nav-link text-light" v-if="node">
                        {{node.name}}
                    </li>
                </ul>
            </div>
        </nav>
        <div v-if="loading">Loading...</div>
        <div id="node-info">
            <!--<node-header :node="node" v-if="node" />-->

            <div class="row" v-if="node_members">
                <div v-for="m of node_members" :key="getUniqueId(m)" class="col col-2">
                    <member-preview v-bind:member="m" />
                </div>
            </div>
            <infinite-loading v-if="node" @infinite="updateMembers" spinner="circles">
                <div slot="no-more" />
            </infinite-loading>
        </div>
        <div id="progress" v-if="progress" v-b-tooltip.hover :title="progress.leaves_n_approved_objects.toLocaleString('en-US') + ' / ' + progress.leaves_n_objects.toLocaleString('en-US')">
            <div :style="{flexGrow: progress.leaves_n_approved_objects}" class="bg-success" />
            <div :style="{flexGrow: progress.leaves_n_objects - progress.leaves_n_approved_objects}" class="bg-danger" />
        </div>
        <div id="decision">
            <b-button id="btn-approve" variant="success" @click.prevent="approve" v-b-tooltip.hover.html title="All members look alike. Approve. <kbd>A</kbd>">Approve</b-button>
            <b-button id="btn-merge" variant="danger" @click.prevent="merge" v-b-tooltip.hover.html title="Members are too dissimilar. Merge into parent. <kbd>M</kbd>">Merge into parent</b-button>
        </div>
        <message-log class="bg-light" :messages="messages" />
        <b-modal ref="doneModal" centered no-fade header-bg-variant="success" title="Approval done">
            <div class="d-block text-center">
                Approval is done for this project.
            </div>
            <footer slot="modal-footer">
                <b-button variant="primary" :to="{name: 'projects'}">Back to projects</b-button>
            </footer>
        </b-modal>
    </div>
</template>

<script>
import axios from "axios";

import InfiniteLoading from "vue-infinite-loading";

import mixins from "@/mixins.js";

import * as api from "@/helpers/api.js";

import MemberPreview from "@/components/MemberPreview.vue";
import MessageLog from "@/components/MessageLog.vue";
import NodeHeader from "@/components/NodeHeader.vue";

export default {
    name: "approve",
    data() {
        return {
            loading: false,
            project: null,
            node: null,
            node_members: [],
            members_url: null,
            progress: null
        };
    },
    created() {
        this.initialize();
    },
    mounted() {
        window.addEventListener("keypress", this.keypress);
    },
    beforeDestroy() {
        window.removeEventListener("keypress", this.keypress);
    },
    components: {
        MemberPreview,
        MessageLog,
        NodeHeader,
        InfiniteLoading
    },
    mixins: [mixins],
    watch: {
        $route: "initialize"
    },
    methods: {
        initialize() {
            this.loading = true;
            this.node = null;
            this.node_members = [];
            this.members_url = null;

            const project_id = parseInt(this.$route.params.project_id);

            var projectLoaded = new Promise(resolve => {
                if (this.project && this.project.project_id == project_id) {
                    console.log("Project was already loaded.");
                    resolve();
                } else {
                    console.log("Loading project.");
                    this.project = null;
                    this.progress = null;
                    api.getProject(project_id, true).then(project => {
                        this.project = project;
                        console.log(project);
                        this.progress = project.progress;
                        resolve();
                    });
                }
            });

            projectLoaded
                .then(() => {
                    // If we already have a node_id, return it
                    if (this.$route.params.node_id) {
                        return parseInt(this.$route.params.node_id);
                    }
                    // ... otherwise get the next node
                    return api
                        .getNextUnapprovedNode(this.project.node_id, true)
                        .then(node_id => {
                            if (node_id == null) {
                                this.$refs.doneModal.show();
                                return null;
                            }
                            const to = {
                                name: "approve",
                                params: {
                                    project_id: project_id,
                                    node_id: node_id
                                }
                            };

                            this.$router.replace(to);
                            return node_id;
                        });
                })
                .then(node_id => {
                    if (node_id == null) {
                        return;
                    }
                    // Avoid double-load by $router.replace(to) -trigger-> initialize
                    if (this.node && this.node.node_id == node_id) {
                        return;
                    }
                    console.log(`Loading node ${node_id}...`);
                    return api.getNode(node_id).then(node => {
                        this.node = node;
                    });
                })
                .then(() => {
                    this.loading = false;
                })
                .catch(e => {
                    this.axiosErrorHandler(e);
                });
        },
        // updateMembers gets called as an infinite loading handler.
        updateMembers($state) {
            // Should members_url be updated (with unique id etc.) on response?
            var updateMembersUrl = false;

            if (!this.members_url) {
                const nodes = this.node.children;
                this.members_url = `/api/nodes/${
                    this.node.node_id
                }/members?objects=${!nodes}&nodes=${nodes}&arrange_by=interleaved&`;
                this.page = 0;
                updateMembersUrl = true;
            }

            var url = `${this.members_url}&page=${this.page}`;

            console.log(`Loading ${url}...`);

            axios
                .get(`${this.members_url}&page=${this.page}`)
                .then(response => {
                    this.node_members = this.node_members.concat(
                        response.data.data
                    );

                    if (updateMembersUrl) {
                        this.members_url = response.data.links.self;
                    }

                    $state.loaded();

                    if (this.page < response.data.meta.last_page) {
                        this.page += 1;
                    } else {
                        $state.complete();
                    }
                })
                .catch(e => {
                    this.axiosErrorHandler(e);
                });
        },
        approve() {
            console.log("Approve");
            api
                .patchNode(this.node.node_id, { approved: true })
                .then(() => {
                    const msg = `Approved ${this.node.node_id}.`;
                    console.log(msg);
                    this.messages.unshift(msg);
                    // Update progress
                    api
                        .getNodeProgress(this.project.node_id, "approve")
                        .then(progress => {
                            this.progress = progress;
                        })
                        .catch(e => {
                            this.axiosErrorHandler(e);
                        });

                    const to = {
                        name: "approve",
                        params: {
                            project_id: this.project.project_id
                        }
                    };

                    this.$router.push(to);
                })
                .catch(e => {
                    this.axiosErrorHandler(e);
                });
        },
        merge() {
            // TODO
            api
                .mergeNodeInto(this.node.node_id, this.node.parent_id)
                .then(() => {
                    this.messages.unshift(`Merged ${this.node.node_id}.`);

                    // Update progress
                    api
                        .getNodeProgress(this.project.node_id, "approve")
                        .then(progress => {
                            this.progress = progress;
                        })
                        .catch(e => {
                            this.axiosErrorHandler(e);
                        });

                    const to = {
                        name: "approve",
                        params: {
                            project_id: this.project.project_id
                        }
                    };

                    this.$router.push(to);
                })
                .catch(e => {
                    this.axiosErrorHandler(e);
                });
        },
        keypress(event) {
            if (
                this.loading ||
                event.altKey ||
                event.ctrlKey ||
                event.metaKey ||
                event.shiftKey
            ) {
                return;
            }
            if (event.key == "a") {
                this.approve();
            } else if (event.key == "m") {
                this.merge();
            }
        }
    }
};
</script>

<style>
#approve {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
}

#node-info {
    flex: 1;
    overflow-y: auto;
}

#node-info .row {
    max-width: 1200px;
    margin: 0 auto;
}

#decision {
    margin: 0 auto;
}

#decision button {
    margin: 0 1em;
}

#progress {
    display: flex;
    flex-wrap: nowrap;
    margin: 0.2em 0;
}

#progress div {
    height: 0.2em;
}
</style>

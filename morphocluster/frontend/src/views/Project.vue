<template>
    <div id="project">
        <nav class="navbar navbar-expand-lg navbar navbar-dark bg-dark">
            <router-link class="navbar-brand" :to="{ name: 'home' }">MorphoCluster</router-link>
            <div class="navbar-collapse" id="navbarNav">
                <ul class="navbar-nav nav-item">
                    <li class="nav-item">
                        <router-link class="nav-link" :to="{ name: 'projects' }">Projects</router-link>
                    </li>
                    <li class="nav-item">
                        <router-link class="nav-link"
                            :to="{ name: 'project', params: { project_id: project.project_id } }">{{
                                project.name }}</router-link>
                    </li>
                </ul>
            </div>
            <dark-mode-control />
        </nav>
        <div class="container">
            <table id="table" style="width=100%">
                <tbody>
                    <tr>
                        <td>Created on</td>
                        <td>{{ project.creation_date }}</td>
                    </tr>
                    <tr>
                        <td>Name</td>
                        <td>{{ project.name }}</td>
                    </tr>
                    <tr>
                        <td>Node_id</td>
                        <td>{{ project.node_id }}</td>
                    </tr>
                    <tr>
                        <td>Project_id</td>
                        <td>{{ project.project_id }}</td>
                    </tr>
                    <tr>
                        <td>Visible</td>
                        <td>{{ project.visible }}</td>
                    </tr>
                </tbody>
            </table>
            <div style="margin: auto ; width: 0; padding-top: 7px ">
                <b-button size="sm" variant="primary" href="" @click.prevent="showSaveModal(project)">
                    Save Project
                </b-button>
            </div>
        </div>
        <b-modal ref="saveModal" lazy centered :title="save_title" @ok="HandleSaveOk">
            <div class="d-block text-center">
                <form @submit.stop.prevent="HandleSaveOk">
                    <b-container fluid>
                        <b-form-row class="my-1">
                            <b-col sm="3">
                                <label for="form_save_name">Name:</label>
                            </b-col>
                            <b-col sm="9">
                                <b-form-input type="text" placeholder="Enter a name for the saved file" id="form_save_name"
                                    v-model="save_slug"></b-form-input>
                            </b-col>
                        </b-form-row>
                        <b-row class="my-3" v-if="save_saving">
                            <b-col sm="12"> Saving... </b-col>
                        </b-row>
                    </b-container>
                </form>
            </div>
        </b-modal>
    </div>
</template>

<script>
import * as api from "@/helpers/api.js";
import axios from "axios";
import { EventBus } from "@/event-bus.js";
import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap";
import DarkModeControl from "@/components/DarkModeControl.vue";

export default {
    name: "ProjectView",
    props: { "project_id": Number },
    components: { DarkModeControl },
    data() {
        return {
            project: null,
            save_slug: "",
            save_title: "",
            save_project_id: null,
            save_saving: false,
        };
    },
    methods: {
        showSaveModal(project) {
            console.log("project", project);
            this.save_slug = project.name;
            this.save_project_id = project.project_id;
            this.save_title = `Save ${project.name} (${this.save_project_id})`;
            this.$refs.saveModal.show();
        },
        HandleSaveOk(evt) {
            evt.preventDefault();

            console.log("Saving", this.save_project_id, "...");
            this.save_saving = true;
            api.saveProject(this.save_project_id).then((result) => {
                console.log("Project saved: " + result["url"]);
                this.save_saving = false;
                this.$nextTick(() => {
                    // Wrapped in $nextTick to ensure DOM is rendered before closing
                    this.$refs.saveModal.hide();
                });
                this.$nextTick(() => {
                    window.open(result["url"] + "?download=1");
                })
            });
        },
    },
    mounted() {
        // Load node info
        axios
            .get(`/api/projects/${this.project_id}`)
            .then(response => {
                this.project = response.data;
                console.log(response.data);

                EventBus.$emit("set-title", this.project.name);
            })
            .catch(e => {
                console.log(e);
            });
    }
};
</script>

<style>
#project {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
}
</style>

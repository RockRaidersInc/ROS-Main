#!/usr/bin/env python
import os
import gi
import subprocess
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

systems = {'Remote': 'rock-desktop.local'}

class ListBoxWindow(Gtk.Window):
    files = []
    selected_files = []
    locations = []
    row_widgets = []
    running = []
    run_page = None
    run_listbox = None

    def __init__(self):
        Gtk.Window.__init__(self, title="ListBox Demo")
        self.set_border_width(10)

        self.nb = Gtk.Notebook()
        self.add(self.nb)

        vbox = Gtk.VBox(spacing=6)

        listbox = Gtk.ListBox()
        listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        vbox.pack_start(listbox, True, True, 0)

        swin = Gtk.ScrolledWindow()
        swin.add_with_viewport(vbox)
        self.nb.append_page(swin, Gtk.Label('Select Files'))


        index = 0
        for root, dirs, files in os.walk('.'):
            if '.git' in root:
                continue
            for f in files:
                if f.endswith('.launch'):
                    row = Gtk.ListBoxRow()
                    hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
                    row.add(hbox)

                    label = Gtk.Label(os.path.join(root, f), xalign=0)
                    check = Gtk.CheckButton()
                    run = Gtk.Button()
                    run.set_label("Run")
                    run.connect("clicked", self.run_single, index)

                    check.connect("toggled", self.toggled_cb, index)
                    self.files.append(os.path.join(root, f))

                    self.locations.append(None)
                    self.row_widgets.append((check, run))
                    self.running.append(False)

                    computer_store = Gtk.ListStore(str)
                    computer_store.append(["Local"])
                    for s in systems:
                        computer_store.append([s])
                    computer_combo = Gtk.ComboBox.new_with_model_and_entry(computer_store)
                    computer_combo.connect("changed", self.on_computer_combo_changed, index)
                    computer_combo.set_entry_text_column(0)
                    computer_combo.set_active(0)


                    hbox.pack_start(label, True, True, 0)
                    hbox.pack_start(computer_combo, False, True, 0)
                    hbox.pack_start(check, False, True, 0)
                    hbox.pack_start(run, False, True, 0)

                    listbox.add(row)

                    index += 1

        self.run_button = Gtk.Button()
        self.run_button.set_label("Run All")
        self.run_button.connect("clicked", self.submit)
        row = Gtk.ListBoxRow()
        row.add(self.run_button)
        listbox.add(row)

        stop_button = Gtk.Button()
        stop_button.set_label("Stop All")
        stop_button.connect("clicked", self.stop_all)
        row = Gtk.ListBoxRow()
        row.add(stop_button)
        listbox.add(row)

        quit_button = Gtk.Button()
        quit_button.set_label("Quit")
        quit_button.connect("clicked", self.quit)
        row = Gtk.ListBoxRow()
        row.add(quit_button)
        listbox.add(row)

    def toggled_cb(self, button, data):
        if button.get_active():
            self.selected_files.append(self.files[data])
        else:
            self.selected_files.remove(self.files[data])

    def on_computer_combo_changed(self, combo, data):
        tree_iter = combo.get_active_iter()
        if tree_iter is not None:
            model = combo.get_model()
            sys = model[tree_iter][0]
            self.locations[data] = sys

    def stop_all(self, button):
        self.stop_launch_files()
        self.run_button.set_sensitive(True)
        self.nb.remove_page(1)
        self.run_listbox = None
        self.run_page = None

    def run_single(self, button, index):
        f = self.files[index]
        self.start_launch_file(index, f)


    def start_launch_file(self, index, f):
        self.running[index] = True
        if self.run_page is None:
            self.run_page = Gtk.VBox(spacing=6)
            self.run_listbox = Gtk.ListBox()

            self.run_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
            self.run_page.pack_start(self.run_listbox, True, True, 0)

            self.nb.append_page(self.run_page, Gtk.Label('Running'))

        row = Gtk.ListBoxRow()
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        row.add(hbox)

        label = Gtk.Label(f, xalign=0)
        stop_button = Gtk.Button()
        stop_button.set_label("Stop")
        stop_button.connect("clicked", self.stop_single, (index, row))
        open_button = Gtk.Button()
        open_button.set_label("Open")
        open_button.connect("clicked", self.open_term, index)


        hbox.pack_start(label, True, True, 0)
        hbox.pack_start(stop_button, False, True, 0)
        hbox.pack_start(open_button, False, True, 0)

        self.run_listbox.add(row)

        machine = self.locations[index]
        fn = (os.path.basename(f)).split('.')[0]
        start_tmux_cmd = 'tmux new -s ' + fn + '_ros_session -d'
        launch_tmux_cmd1 = 'tmux send-keys -l -t ' + fn + '_ros_session '
        launch_tmux_cmd2 = 'tmux send-keys -t ' + fn + '_ros_session '
        ros_cmd = 'roslaunch Space ' + f + ' Enter'

        if machine == 'Local':
            # Run file locally
            subprocess.call(start_tmux_cmd, shell=True)
            subprocess.call(launch_tmux_cmd2 + 'source Space ../devel/setup.bash Enter', shell=True)
            subprocess.call(launch_tmux_cmd2 + ros_cmd, shell=True)
        else:
            # Run on remote machine
            ssh_cmd = 'ssh rock@' + systems[machine] + ' '
            remote_cmd = 'roslaunch Space URC/src/' + f + ' Enter'

            subprocess.call(ssh_cmd + start_tmux_cmd, shell=True)
            subprocess.call(ssh_cmd + launch_tmux_cmd2 + 'source Space URC/devel/setup.bash Enter', shell=True)
            subprocess.call(ssh_cmd + launch_tmux_cmd2 + 'source Space URC/src/launchscrips/export_remote_ros_vars.sh Enter', shell=True)
            subprocess.call(ssh_cmd + launch_tmux_cmd2 + remote_cmd, shell=True)

            print(ssh_cmd + start_tmux_cmd)
        
        self.row_widgets[index][0].set_active(True)
        self.row_widgets[index][1].set_sensitive(False)
        self.nb.show_all()

    def submit(self, button):
        print('Submitted')
        for index, f in enumerate(self.files):
            if f not in self.selected_files:
                continue 
            if self.running[index]:
                continue
            self.start_launch_file(index, f)

            

    def quit(self, button):
        print('Quitting')
        self.stop_launch_files()
        Gtk.main_quit()

    def open_term(self, button, data):
        f = self.files[data]
        fn = (os.path.basename(f)).split('.')[0]
        machine = self.locations[data]        
        if machine == 'Local':
            subprocess.call('gnome-terminal --command="tmux a -t ' + fn + '_ros_session"', shell=True)
            #print('Local:')
            #print('gnome-terminal --command="tmux a -t ' + fn + '_ros_session"')
        else:
            #subprocess.call('gnome-terminal --command="ssh rock@' + systems[machine] + ' tmux a -t ' + fn + '_ros_session"', shell=True)
            subprocess.call('gnome-terminal --command="ssh rock@' + systems[machine], shell=True)
            #print('Remote:')
            #print('gnome-terminal --command="ssh rock@' + systems[machine] + ' tmux a -t ' + fn + '_ros_session"')


    def stop_single(self, button, data):
        f = self.files[data[0]]
        self.stop_launch_file(f, data[0])

        self.run_listbox.remove(data[1])
        print(len(self.run_listbox.get_children()))
        if len(self.run_listbox.get_children()) == 0:
            self.run_button.set_sensitive(True)
            self.nb.remove_page(1)
            self.run_listbox = None
            self.run_page = None

            

    def stop_launch_file(self, f, index):
        self.running[index] = False
        machine = self.locations[index]
        fn = (os.path.basename(f)).split('.')[0]
        tmux_cmd = 'tmux kill-session -t ' + fn + '_ros_session'

        self.row_widgets[index][1].set_sensitive(True)

        if machine == 'Local':
            # Run file locally
            subprocess.call(tmux_cmd, shell=True)
        else:
            # Run on remote machine
            ssh_cmd = 'ssh rock@' + systems[machine] + ' '
            subprocess.call(ssh_cmd + tmux_cmd, shell=True)
        

    def stop_launch_files(self):
        for index, f in enumerate(self.files):
            if f not in self.selected_files:
                continue 
            self.stop_launch_file(f, index)


win = ListBoxWindow()
win.connect("delete-event", Gtk.main_quit)
win.show_all()
Gtk.main()

'''

for root, dirs, files in os.walk('.'):
    if '.git' in root:
        continue
    for f in files:
        if f.endswith('.launch'):
            print(os.path.join(root, f))

'''

#ifndef button_test_cpp__BUTTON_TEST_H
#define button_test_cpp__BUTTON_TEST_H

#include <rqt_gui_cpp/plugin.h>
//#include <button_test_cpp/ui_button_test.h>
#include <QWidget>

namespace button_test_cpp {

class ButtonTest
  : public rqt_gui_cpp::Plugin
{
  Q_OBJECT
public:
  ButtonTest();
  virtual void initPlugin(qt_gui_cpp::PluginContext& context);
  virtual void shutdownPlugin();
  virtual void saveSettings(qt_gui_cpp::Settings& plugin_settings, 
    qt_gui_cpp::Settings& instance_settings) const;
  virtual void restoreSettings(const qt_gui_cpp::Settings& plugin_settings, 
    const qt_gui_cpp::Settings& instance_settings);

  // Comment in to signal that the plugin has a way to configure it
  //bool hasConfiguration() const;
  //void triggerConfiguration();
private:
  //Ui::ButtonTestWidget ui_;
  QWidget* widget_;
};
} // namespace
#endif // button_test__BUTTON_TEST_H
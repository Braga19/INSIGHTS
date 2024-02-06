from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, IntegerField, DateField, SubmitField, RadioField
from wtforms.validators import DataRequired, NumberRange

class ReviewForm(FlaskForm):
    start_date = DateField('Start Date',format='%Y-%m-%d', validators=[DataRequired()])
    end_date = DateField('End Date', format='%Y-%m-%d', validators=[DataRequired()])
    output_type = RadioField('Choose Distribution', choices=[('daily','Daily'),('weekly','Weekly'),('monthly','Monthly')], validators=[DataRequired()])
    submit = SubmitField('Submit')

class KeywordsForm(FlaskForm):
    form = SelectField('Form', choices=[('word', 'Word'), ('pair', 'Pair')], validators=[DataRequired()])
    polar = SelectField('Polarity', choices=[('positive', 'Positive'), ('negative', 'Negative')], validators=[DataRequired()])
    platform = SelectField('Platform', choices=[('ios', 'iOS'), ('android', 'Android')], validators=[DataRequired()])
    n = IntegerField('N', validators=[NumberRange(min=5, max=20)])
    date = StringField('Date', validators=[DataRequired()])
    submit = SubmitField('Submit')

class DistributionForm(FlaskForm):
    start_date = DateField('Start Date', format='%Y-%m-%d', validators=[DataRequired()])
    end_date = DateField('End Date', format='%Y-%m-%d', validators=[DataRequired()])
    submit = SubmitField('Create Report')

class SearchPatternForm(FlaskForm):
    pattern = StringField('Pattern', validators=[DataRequired()])
    platform = SelectField('Platform', choices=[('ios', 'iOS'), ('android', 'Android')], validators=[DataRequired()])
    submit = SubmitField('Submit')
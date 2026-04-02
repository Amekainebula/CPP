#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
int cnt = 0;
bool is_huiwen(string s)
{
    int n = s.length();
    for (int i = 0; i < n / 2; i++)
    {
        if (s[i] != s[n - 1 - i])
        {
            return false;
        }
    }
    return true;
}
int temp;
void change(string &s)
{
    int n = s.length();
    if (n % 2 == 1)
    {
        if (s[n / 2] == '0')
        {
            s[n / 2] = '1';
            return;
        }
    }
    for (int i = 0; i < n / 2; i++)
    {
        if (s[i] != s[n - 1 - i])
        {
            cnt--;
            temp++;
        }
    }
}
void solve()
{
    int n;
    cin >> n;
    string s;
    cin >> s;
    for (int i = 0; i < s.length(); i++)
        if (s[i] == '0')
            cnt++;
    int now = 0;
    if (is_huiwen(s))
    {
        if (s.length() % 2 == 1 && s[s.length() / 2] == '0')
        {
            if (cnt > 1)
                cout << "F\n";
            else
                cout << "L\n";
        }
        else
        {
            cout << "L\n";
        }
    }
    else
    {
        change(s);
        if (cnt == 0)
            cout << "F\n";
        else
        {
            if (temp == 1)
                cout << "L\n";
            else
                cout << "F\n";
        }
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}
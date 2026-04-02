#include <bits/stdc++.h>
#define int long long
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define endl '\n' 
#define yes cout << "YES" << endl
#define no cout << "NO" << endl
#define yess cout << "Yes"<< endl
#define noo cout << "No" << endl
using namespace std;
void solve()
{
    int n;
    cin >> n;
    string s;
    cin >> s;
    char now = 'a';
    string t;
    int ans = 0;
    for (int i = 0; i < n; i++)
    {
        if (now == 'a')
        {
            t += s[i];
            now = s[i];
        }
        if (now != s[i])
        {
            t += s[i];
            now = s[i];
        }
    }
    int cnt = 0;
    if (t[0] == '1') ans++, cnt++;
    for (int i = 1; i < sz(t); i++)
    {
        if (t[i] == '1') cnt++;
    }
    cout << ans + (cnt - 1) * 2 << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}
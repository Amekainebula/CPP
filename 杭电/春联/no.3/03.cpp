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
int num(int l,int r,string s,vector<int> &vis)
{
    int nn=0;
    for(int i=l;i<=r;i++)
    {
        if(!vis[i])
        {
            vis[i]=1;
            if(s[i]>=1&&s[i]<=9)
            nn=nn*10+s[i]-'0';
            else if(s[i]=='+')
            {
                
            }
        }
    }
}
void solve()
{
    string s;
    cin >> s;
    stack<int> st;
    queue<int> q;
    vector<int>vis(s.size()+10, 0);
    for (int i = 0; i < s.size(); i++)
    {
        if (s[i] == '(')
            st.push(i);
        else if (s[i] == ')')
            q.push(i);
    }
    double now=0;
    while (!q.empty()&&!st.empty())
    {
        int l = st.top();
        int r = q.front();
        st.pop();
        q.pop();
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